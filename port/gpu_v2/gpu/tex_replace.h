// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Optional guest texture trace / dump / replacement for the renderer v2 texture cache, written in
// the same spirit as this port's engine-level diagnostics (TEXTURE_GC, ISS_BIG_UPLOAD_LOG): every
// switch is off by default (defaults.env) and costs nothing when off.
//
//   ISS_TEX_TRACE=1     log every guest-to-GPU image upload (first-seen variants + a count)
//   ISS_TEX_DUMP=1      also write one file per first-seen image into the dump folder
//   ISS_TEX_REPLACE=1   apply replacement files onto the uploads they match
//
// Dump folder: tex_dump\ next to the engine (env ISS_TEX_DUMP_DIR), manifest.csv alongside.
// Replacement folder: tex_replace\ (env ISS_TEX_REPLACE_DIR).
//
// The dump file name is also the replacement file name (a dumped PNG with redrawn pixels is copied
// into tex_replace unchanged - the folders and the file identity carry everything else):
//   <addr16hex>_<WxH>_f<fmt>_t<tile>_<n>.png   formats decodeable to 8-bit channels (see PngType)
//   <addr16hex>_<WxH>_f<fmt>_t<tile>_<n>.raw   everything else (tiled, block-compressed, 16bpp)
// <n> counts first-seen variants that shared an address (arena re-loads land higher). manifest.csv
// records the first-256-byte guest hash of every dump; a replacement checks it before applying, so
// a stale address that the game re-used for another texture is skipped instead of repainted.
//
// A .raw replacement must hold exactly info.guest_size bytes (the layout of the detiled upload
// staging this pass writes). A .png replacement is decoded and rebuilt through the source pixel
// format row by row, mip chain regenerated. The blit itself is the very Runtime::UploadImage call
// the guest upload just used, so layout, barriers and staging handling remain the engine's own.
//
// One honesty note: a .raw dump of a TILED image holds guest-side swizzled bytes while the staging
// expects the linear layout, so replacing a tiled texture from its dump is a byte experiment, not
// an image edit. The UI prompt atlases this feature targets are linear textures, for which the
// dump/replace cycle is exact end to end (and .png handles it).

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <xxhash.h>

#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/texture_cache/image.h"

// stb declarations only: the single implementation TU is shadps4/common/stb.cpp, compiled with
// STBI_ONLY_PNG and STBI_NO_STDIO, so only memory-based stbi functions exist as symbols here.
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "common/stb.h"

namespace iss::tex_replace {

namespace fs = std::filesystem;

namespace {

struct Settings {
    bool trace = false;
    bool dump = false;
    bool replace = false;
    u32 dump_limit = 2048;
    u64 dump_max_bytes = u64{16} << 20;
    fs::path dump_dir;
    fs::path replace_dir;
};

// One replacement source file, fully loaded at engine start (the folder holds a handful of small
// images; the upload path only does map lookups, a fingerprint check and a memcpy).
struct ReplEntry {
    u64 address{};
    u32 width{};
    u32 height{};
    u32 format{};
    u32 tile{};
    u32 files{};
    u64 file_bytes{};
    u64 dump_fp{}; // first-256-byte guest hash recorded in manifest.csv, for the content guard
    bool is_png{false};
    std::vector<u8> raw;  // .raw: exact staging bytes
    std::vector<u8> rgba; // .png: decoded RGBA8 (width * height * 4)
    std::string name;
};

std::mutex g_mutex;
std::atomic<bool> g_ready{false};
Settings g_settings;
std::unordered_map<u64, std::vector<ReplEntry>> g_replacements;
std::unordered_map<std::string, u64> g_repl_fp; // manifest name -> dumped fingerprint
std::unordered_set<u64> g_trace_firsts;
std::unordered_set<u64> g_dump_firsts;
std::unordered_map<u64, u32> g_dump_seq; // variants of one address
std::unordered_set<std::string> g_warned;
u32 g_dumps = 0;
std::atomic<u64> g_trace_uploads{0};

bool FlagOn(const char* name) {
    const char* p = std::getenv(name);
    return p && p[0] && p[0] != '0';
}

void WarnOnce(const std::string& what, const char* fmt, ...) {
    if (!g_warned.insert(what).second) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    std::fputs("[tex_replace] ", stderr);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// Without ISS_TEX_*_DIR the relative paths follow the engine's working directory; an absolute
// ISS_TEX_REPLACE_DIR / ISS_TEX_DUMP_DIR line in defaults.env removes all doubt.
fs::path DirSetting(const char* name, const char* fallback) {
    const char* p = std::getenv(name);
    if (p && p[0]) {
        return fs::path(p);
    }
    return fs::path(fallback);
}

Settings LoadSettings() {
    Settings s;
    s.trace = FlagOn("ISS_TEX_TRACE");
    s.dump = FlagOn("ISS_TEX_DUMP");
    s.replace = FlagOn("ISS_TEX_REPLACE");
    if (const char* p = std::getenv("ISS_TEX_DUMP_LIMIT"); p && p[0]) {
        const auto v = std::strtoul(p, nullptr, 10);
        if (v != 0) {
            s.dump_limit = static_cast<u32>(v);
        }
    }
    if (const char* p = std::getenv("ISS_TEX_DUMP_MAX_MB"); p && p[0]) {
        const auto v = std::strtoul(p, nullptr, 10);
        if (v != 0) {
            s.dump_max_bytes = u64{v} << 20;
        }
    }
    s.dump_dir = DirSetting("ISS_TEX_DUMP_DIR", "tex_dump");
    s.replace_dir = DirSetting("ISS_TEX_REPLACE_DIR", "tex_replace");
    return s;
}

// The file names are the contract between the dump and the replacement folders; parse them back.
// <addr x16 hex>_<W>x<H>_f<F>_t<T>_<n>.(png|raw)
bool ParseHex16(const char*& p, u64& out) {
    u64 v = 0;
    for (int i = 0; i < 16; ++i) {
        const char c = *p;
        int d;
        if (c >= '0' && c <= '9') {
            d = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            d = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            d = c - 'A' + 10;
        } else {
            return false;
        }
        v = (v << 4) | static_cast<u64>(d);
        ++p;
    }
    out = v;
    return true;
}

bool ParseUint(const char*& p, u64& out) {
    if (*p < '0' || *p > '9') {
        return false;
    }
    u64 v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + static_cast<u64>(*p - '0');
        ++p;
    }
    out = v;
    return true;
}

bool ParseName(const std::string& name, ReplEntry& e) {
    const char* p = name.c_str();
    u64 addr = 0, w = 0, h = 0, fmt = 0, tile = 0, n = 0;
    if (!ParseHex16(p, addr) || *p++ != '_' || !ParseUint(p, w) || *p++ != 'x' ||
        !ParseUint(p, h) || *p++ != '_' || *p++ != 'f' || !ParseUint(p, fmt) || *p++ != '_' ||
        *p++ != 't' || !ParseUint(p, tile) || *p++ != '_' || !ParseUint(p, n) || *p++ != '.') {
        return false;
    }
    std::string ext(p);
    e.address = addr;
    e.width = static_cast<u32>(w);
    e.height = static_cast<u32>(h);
    e.format = static_cast<u32>(fmt);
    e.tile = static_cast<u32>(tile);
    e.files = static_cast<u32>(n);
    e.is_png = ext.compare(0, 3, "png") == 0;
    return true;
}

void LoadManifest() {
    const fs::path file = g_settings.dump_dir / "manifest.csv";
    std::FILE* f = std::fopen(file.string().c_str(), "r");
    if (!f) {
        return;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        char name[300];
        u64 fp = 0;
        if (std::sscanf(line, "%299[^,],%16llx", name, &fp) == 2) {
            g_repl_fp[name] = fp;
        }
    }
    std::fclose(f);
}

u64 GuestFingerprint(const VideoCore::Image& image) {
    const auto* guest = std::bit_cast<const u8*>(image.info.guest_address);
    if (!guest || image.info.guest_size == 0) {
        return 0;
    }
    const u64 n = std::min<u64>(image.info.guest_size, 256);
    return XXH3_64bits(guest, n);
}

void LoadReplacements() {
    std::error_code ec;
    if (!fs::exists(g_settings.replace_dir, ec)) {
        WarnOnce("no-folder", "ISS_TEX_REPLACE=1 but no replacement folder at '%s'",
                 g_settings.replace_dir.string().c_str());
        return;
    }
    LoadManifest();
    u32 loaded = 0;
    for (const auto& item : fs::directory_iterator(g_settings.replace_dir, ec)) {
        if (ec) {
            break;
        }
        if (!item.is_regular_file(ec)) {
            continue;
        }
        const std::string name = item.path().filename().string();
        const bool is_png = name.size() > 4 && name.compare(name.size() - 4, 4, ".png") == 0;
        const bool is_raw = name.size() > 4 && name.compare(name.size() - 4, 4, ".raw") == 0;
        if (!is_png && !is_raw) {
            continue;
        }
        ReplEntry e{};
        e.name = name;
        e.is_png = is_png;
        if (!ParseName(name, e)) {
            WarnOnce("bad-name-" + name, "skipping '%s': name is not in the dump format", name.c_str());
            continue;
        }
        std::FILE* f = std::fopen(item.path().string().c_str(), "rb");
        if (!f) {
            continue;
        }
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<u8> bytes(size > 0 ? static_cast<size_t>(size) : 0);
        const size_t got = bytes.empty() ? 0 : std::fread(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        bytes.resize(got);
        if (bytes.empty() || bytes.size() > (u64{64} << 20)) {
            WarnOnce("size-" + name, "skipping '%s': %zu bytes, malformed or oversized", name.c_str(),
                     bytes.size());
            continue;
        }
        if (is_png) {
            int w = 0, h = 0, comp = 0;
            if (u8* decoded = stbi_load_from_memory(bytes.data(),
                                                    static_cast<int>(bytes.size()), &w, &h,
                                                    &comp, 4)) {
                e.width = static_cast<u32>(w);
                e.height = static_cast<u32>(h);
                e.rgba.assign(decoded, decoded + static_cast<size_t>(w) * h * 4);
                stbi_image_free(decoded);
            } else {
                WarnOnce("bad-png-" + name, "skipping '%s': unreadable PNG (%s)", name.c_str(),
                         stbi_failure_reason() ? stbi_failure_reason() : "decode failed");
                continue;
            }
        } else {
            e.raw = std::move(bytes);
            e.file_bytes = e.raw.size();
        }
        if (const auto it = g_repl_fp.find(name); it != g_repl_fp.end()) {
            e.dump_fp = it->second;
        }
        g_replacements[e.address].push_back(std::move(e));
        ++loaded;
    }
    std::fprintf(stderr, "TEXREP %u replacement(s) loaded from '%s'%s\n", loaded,
                 g_settings.replace_dir.string().c_str(), loaded ? "" : " (nothing to apply)");
    std::fflush(stderr);
}

// Formats a dump/replacement can travel as PNG (RGBA8 in, same per-pixel channels out).
bool PngType(u32 format) {
    switch (format) {
    case 37: // VK_FORMAT_R8G8B8A8_UNORM
    case 38: // VK_FORMAT_R8G8B8A8_SRGB
    case 43: // VK_FORMAT_B8G8R8A8_UNORM
    case 50: // VK_FORMAT_B8G8R8A8_SRGB
    case 29: // VK_FORMAT_B8G8R8_UNORM
    case 8:  // VK_FORMAT_A8B8G8R8_UNORM_PACK32
    case 9:  // VK_FORMAT_R8_UNORM
    case 16: // VK_FORMAT_R8G8_UNORM
        return true;
    default:
        return false;
    }
}

// Memory byte order of the supportedformats: for each of R,G,B,A the index of its byte in the
// pixel, or 255 when the channel does not exist (opaque). Returns nullptr when the bytes are
// already RGBA (the PNG's own order).
const u8* ChannelMap(u32 format) {
    static constexpr u8 bgra[4] = {2, 1, 0, 3}; // B8G8R8A8(_SRGB)
    static constexpr u8 bgr[4] = {2, 1, 0, 255}; // B8G8R8
    static constexpr u8 abgr[4] = {3, 2, 1, 0}; // A8B8G8R8_PACK32
    static constexpr u8 r[4] = {0, 255, 255, 255}; // R8
    static constexpr u8 rg[4] = {0, 1, 255, 255}; // R8G8
    switch (format) {
    case 43:
    case 50:
        return bgra;
    case 29:
        return bgr;
    case 8:
        return abgr;
    case 9:
        return r;
    case 16:
        return rg;
    default: // 37/38: R8G8B8A8(_SRGB) is RGBA as decoded
        return nullptr;
    }
}

void WriteDumpPng(const fs::path& file, u32 width, u32 height, u32 format, const u8* guest,
                  u64 row_bytes) {
    std::vector<u8> rgba(static_cast<size_t>(width) * height * 4);
    const u8* map = ChannelMap(format);
    for (u32 y = 0; y < height; ++y) {
        const u8* src = guest + u64{y} * row_bytes;
        u8* dst = rgba.data() + u64{y} * width * 4;
        for (u32 x = 0; x < width; ++x) {
            const u8* px = src + u64{x} * 4;
            u8* out = dst + u64{x} * 4;
            if (!map) {
                out[0] = px[0];
                out[1] = px[1];
                out[2] = px[2];
                out[3] = px[3];
            } else {
                out[0] = px[map[0]];
                out[1] = px[map[1]];
                out[2] = px[map[2]];
                out[3] = map[3] == 255 ? 255 : px[map[3]];
            }
        }
    }
    std::vector<u8> png;
    const auto writer = +[](void* ctx, void* data, int size) {
        auto* out = static_cast<std::vector<u8>*>(ctx);
        const auto* bytes = static_cast<const u8*>(data);
        out->insert(out->end(), bytes, bytes + size);
    };
    if (!stbi_write_png_to_func(writer, &png, static_cast<int>(width), static_cast<int>(height),
                                4, rgba.data(), static_cast<int>(width) * 4)) {
        std::fprintf(stderr, "TEXDUMP png write failed %s\n", file.string().c_str());
        return;
    }
    std::FILE* f = std::fopen(file.string().c_str(), "wb");
    if (!f) {
        return;
    }
    std::fwrite(png.data(), 1, png.size(), f);
    std::fclose(f);
}

void WriteDumpRaw(const fs::path& file, const u8* guest, u64 bytes) {
    std::FILE* f = std::fopen(file.string().c_str(), "wb");
    if (!f) {
        return;
    }
    std::fwrite(guest, 1, static_cast<size_t>(bytes), f);
    std::fclose(f);
}

std::string DumpFileName(const VideoCore::Image& image, u32 seq, bool png) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%016llx_%ux%u_f%u_t%u_%u.%s",
                  (unsigned long long)image.info.guest_address, image.info.size.width,
                  image.info.size.height, static_cast<u32>(image.info.pixel_format),
                  static_cast<u32>(image.info.tile_mode), seq, png ? "png" : "raw");
    return std::string(buf);
}

void DumpImage(const VideoCore::Image& image) {
    const auto& info = image.info;
    if (g_dumps >= g_settings.dump_limit || info.guest_size > g_settings.dump_max_bytes) {
        return;
    }
    // First-seen key: address and the dumped fingerprint. Re-uploads of the same bytes are noise;
    // a different fingerprint at the same address is a new texture (arena re-load) - next file.
    const u64 fp = GuestFingerprint(image);
    const u64 key = info.guest_address * 0x9E3779B97F4A7C15ULL ^ (fp ? fp : u64{1});
    auto [it, inserted] = g_dump_firsts.insert(key);
    if (!inserted) {
        return;
    }
    (void)it;
    ++g_dumps;
    const u32 seq = ++g_dump_seq[info.guest_address];
    const bool png =
        !info.props.is_tiled && !info.props.is_block && PngType(static_cast<u32>(info.pixel_format));
    const std::string name = DumpFileName(image, seq, png);
    std::error_code ec;
    fs::create_directories(g_settings.dump_dir, ec);
    if (ec) {
        WarnOnce("no-dump-dir", "cannot create dump folder '%s': %s",
                 g_settings.dump_dir.string().c_str(), ec.message().c_str());
        return;
    }
    const fs::path file = g_settings.dump_dir / name;
    const auto* guest = std::bit_cast<const u8*>(info.guest_address);
    if (png) {
        WriteDumpPng(file, info.size.width, info.size.height,
                     static_cast<u32>(info.pixel_format), guest,
                     u64{info.mips_layout[0].pitch ? info.mips_layout[0].pitch : info.size.width} *
                         (info.num_bits / 8));
    } else {
        WriteDumpRaw(file, guest, info.guest_size);
    }
    if (std::FILE* mf = std::fopen((g_settings.dump_dir / "manifest.csv").string().c_str(), "a")) {
        std::fprintf(mf, "%s,%016llx,%llu\n", name.c_str(), (unsigned long long)fp,
                     (unsigned long long)info.guest_size);
        std::fclose(mf);
    }
    std::fprintf(stderr,
                 "TEXDUMP %s %ux%u fmt=%u(%s) tile=%u bytes=%u mips=%u layers=%u tiled=%d\n",
                 name.c_str(), info.size.width, info.size.height,
                 static_cast<u32>(info.pixel_format), vk::to_string(info.pixel_format).c_str(),
                 static_cast<u32>(info.tile_mode), u32(info.guest_size), u32(info.resources.levels),
                 u32(info.resources.layers), int(info.IsTiled()));
    std::fflush(stderr);
}

void TraceImage(const VideoCore::Image& image) {
    const u64 fp = GuestFingerprint(image);
    const u64 key = image.info.guest_address * 0x9E3779B97F4A7C15ULL ^ (fp ? fp : u64{1});
    if (g_trace_firsts.size() > 20000) {
        return; // saturation guard: the log already carries the atlas
    }
    if (g_trace_firsts.insert(key).second) {
        const auto& info = image.info;
        std::fprintf(stderr,
                     "TEXTRACE first addr=%llx size=%u %ux%ux%d fmt=%u(%s) tile=%u mips=%u "
                     "layers=%u samples=%u fp=%llx\n",
                     (unsigned long long)info.guest_address, u32(info.guest_size),
                     info.size.width, info.size.height, info.size.depth,
                     static_cast<u32>(info.pixel_format), vk::to_string(info.pixel_format).c_str(),
                     static_cast<u32>(info.tile_mode), u32(info.resources.levels),
                     u32(info.resources.layers), u32(info.num_samples), (unsigned long long)fp);
    }
    if ((g_trace_uploads.fetch_add(1, std::memory_order_relaxed) % 600) == 599) {
        std::fprintf(stderr, "TEXTRACE uploads=%llu\n",
                     (unsigned long long)g_trace_uploads.load(std::memory_order_relaxed));
        std::fflush(stderr);
    }
}

// Rebuild a PNG replacement through the source pixel format's mip layout; the mip chain is
// re-decimated from the source image (mips share one staging buffer, the offsets stand).
void PngToNativeLayout(const VideoCore::Image& image, const std::vector<u8>& rgba, std::vector<u8>& out) {
    const auto& info = image.info;
    const u8 bpp = static_cast<u8>(info.num_bits / 8);
    const u8* map = ChannelMap(static_cast<u32>(info.pixel_format));
    for (u32 m = 0; m < info.resources.levels; ++m) {
        const auto& mip = info.mips_layout[m];
        const u32 w = std::max(info.size.width >> m, 1u);
        const u32 h = std::max(info.size.height >> m, 1u);
        const u32 pitch = mip.pitch ? mip.pitch : w;
        for (u32 y = 0; y < h; ++y) {
            u8* dst = out.data() + u64{mip.offset} + u64{y} * pitch * bpp;
            for (u32 x = 0; x < w; ++x) {
                const u32 sx = std::min(x << m, info.size.width - 1);
                const u32 sy = std::min(y << m, info.size.height - 1);
                const u8* px = rgba.data() + (u64{sy} * info.size.width + sx) * 4;
                if (!map) {
                    dst[x * bpp + 0] = px[0];
                    dst[x * bpp + 1] = px[1];
                    dst[x * bpp + 2] = px[2];
                    if (bpp > 3) {
                        dst[x * bpp + 3] = px[3];
                    }
                } else {
                    dst[x * bpp + 0] = px[map[0]];
                    dst[x * bpp + 1] = px[map[1]];
                    dst[x * bpp + 2] = px[map[2]];
                    if (bpp > 3) {
                        dst[x * bpp + 3] = map[3] == 255 ? 255 : px[map[3]];
                    }
                }
            }
        }
    }
}

void ApplyReplacement(const VideoCore::Image& image, Vulkan::Runtime& runtime,
                      const std::span<const vk::BufferImageCopy> upload_copies) {
    const auto& info = image.info;
    const auto it = g_replacements.find(info.guest_address);
    if (it == g_replacements.end() || it->second.empty()) {
        return;
    }
    const ReplEntry* pick = nullptr;
    for (const auto& e : it->second) {
        if (e.width != info.size.width || e.height != info.size.height) {
            continue;
        }
        if (e.dump_fp != 0 && e.dump_fp != GuestFingerprint(image)) {
            // The address now holds other bytes than the dumped texture had. Leave it alone.
            WarnOnce("guarded-" + e.name,
                     "'%s': guest bytes changed since the dump, skipping", e.name.c_str());
            continue;
        }
        if (e.is_png) {
            if (info.props.is_tiled || info.props.is_block ||
                !PngType(static_cast<u32>(info.pixel_format))) {
                WarnOnce("png-fmt-" + e.name,
                         "'%s': this upload's format cannot take a PNG, widen to a .raw dump "
                         "instead", e.name.c_str());
                continue;
            }
        } else {
            if (e.file_bytes != info.guest_size) {
                WarnOnce("raw-size-" + e.name, "'%s': %llu bytes but this staging is %u",
                         e.name.c_str(), (unsigned long long)e.file_bytes, u32(info.guest_size));
                continue;
            }
            if (e.format != 0 && e.format != static_cast<u32>(info.pixel_format)) {
                WarnOnce("raw-fmt-" + e.name, "'%s': the format at this address changed, skipping",
                         e.name.c_str());
                continue;
            }
        }
        pick = &e;
        break;
    }
    if (!pick) {
        return;
    }

    // The detiled staging covers exactly info.guest_size bytes of the image layout.
    std::vector<u8> staging_native(info.guest_size);
    if (pick->is_png) {
        PngToNativeLayout(image, pick->rgba, staging_native);
    } else {
        std::memcpy(staging_native.data(), pick->raw.data(), staging_native.size());
    }

    const auto staging = runtime.GetStagingPool().Request(staging_native.size(),
                                                          VideoCore::MemoryType::HostUncached,
                                                          16);
    if (!staging.mapped) {
        return;
    }
    std::memcpy(staging.mapped, staging_native.data(), staging_native.size());
    staging.Flush();

    std::vector<vk::BufferImageCopy> repl_copies(upload_copies.begin(), upload_copies.end());
    for (auto& copy : repl_copies) {
        copy.bufferOffset += staging.offset;
    }
    runtime.UploadImage(const_cast<VideoCore::Image*>(&image), staging.buffer, repl_copies);
    std::fprintf(stderr, "TEXREP applied %s addr=%llx uid=%llu\n", pick->name.c_str(),
                 (unsigned long long)info.guest_address, (unsigned long long)image.image_uid);
    std::fflush(stderr);
}

} // namespace

// Called right after the normal guest-to-GPU blit of an image. With every switch off this costs
// one relaxed atomic load.
template <typename CopyRange>
void OnImageUpload(VideoCore::Image& image, Vulkan::Runtime& runtime, const CopyRange& upload_copies) {
    if (!g_ready.load(std::memory_order_acquire)) {
        std::scoped_lock lock{g_mutex};
        if (!g_ready.load(std::memory_order_relaxed)) {
            g_settings = LoadSettings();
            std::error_code ec;
            if (g_settings.dump) {
                fs::create_directories(g_settings.dump_dir, ec);
            }
            if (g_settings.replace) {
                LoadReplacements();
            }
            g_ready.store(true, std::memory_order_release);
        }
    }
    if (g_settings.trace || g_settings.dump) {
        std::fprintf(stderr,
                     "TEXTRACE on trace=%d dump=%d replace=%d dump_dir='%s' replace_dir='%s'\n",
                     int(g_settings.trace), int(g_settings.dump), int(g_settings.replace),
                     g_settings.dump_dir.string().c_str(),
                     g_settings.replace_dir.string().c_str());
        std::fflush(stderr);
    }
    if (!(g_settings.trace || g_settings.dump || g_settings.replace)) {
        return;
    }
    const auto& info = image.info;
    if (info.guest_address == 0 || info.guest_size == 0 ||
        info.pixel_format == vk::Format::eUndefined || info.props.is_depth) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    if (g_settings.trace) {
        TraceImage(image);
    }
    if (g_settings.dump) {
        DumpImage(image);
    }
    if (g_settings.replace) {
        std::span<const vk::BufferImageCopy> copies(upload_copies.data(), upload_copies.size());
        ApplyReplacement(image, runtime, copies);
    }
}

} // namespace iss::tex_replace
