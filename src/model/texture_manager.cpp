
#include "texture_2d.h"

#include "texture_manager.h"

namespace fs = std::filesystem;

TextureManager::TextureManager(Context& context)
    : context_(context)
{
    std::unique_ptr<Texture2D> texture = std::make_unique<Texture2D>("assets", "vulkan_cloth_rgba.ktx", context);
    textures_.push_back(std::move(texture));

    ConvertPbrPngsInFolderToKtx("assets/worm");
}

uint32_t TextureManager::CreateTexture2D(std::string path, std::string keyword)
{
    auto CreateTexture = [&](std::string path, std::string filename)
        {
            std::string texPath = path + "/" + filename;
            std::cout << "Load " << texPath << std::endl;
            fs::path p(texPath);
            std::unique_ptr<Texture2D> texture = std::make_unique<Texture2D>(path, filename, context_);
            uint32_t idx = textures_.size();
            textures_.push_back(std::move(texture));
            return idx;
        };

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_regular_file()) continue;

        std::filesystem::path p(entry.path());

        //std::cout << "전체 경로: " << p << "\n";
        //std::cout << "파일 이름: " << p.filename() << "\n";
        //std::cout << "확장자: " << p.extension() << "\n";
        //std::cout << "부모 폴더: " << p.parent_path() << "\n";
        //std::cout << "이름(확장자 제외): " << p.stem() << "\n";

        std::string parentPath = p.parent_path().string();
        std::string filename = p.filename().string();

        if (p.extension() != ".ktx") continue;

        if (filename.find(keyword) != std::string::npos)
            return CreateTexture(parentPath, filename);

    }
    
    return 0;
}

TextureManager::~TextureManager()
{

}

// 개별 PNG를 KTX로 변환
void TextureManager::CreateKtxFromPng(const fs::path& pngPath, const fs::path& ktxPath)
{
    std::string filename = pngPath.filename().string();
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });

    bool isAlbedo =
        lower.find("albedo") != std::string::npos ||
        lower.find("basecolor") != std::string::npos ||
        lower.find("base_color") != std::string::npos;

    std::string fmt;
    std::string tf;

    if (isAlbedo) {
        // 색상 텍스쳐
        fmt = "R8G8B8A8_SRGB";
        tf = "srgb";
    }
    else {
        // 데이터 텍스쳐 (normal, roughness, metallic ...)
        fmt = "R8G8B8A8_UNORM";
        tf = "linear";
    }

    std::string cmd = "ktx create "
        "--format " + fmt + " "
        "--assign-tf " + tf + " "
        "--assign-primaries bt709 "
        "--generate-mipmap "  // 혹은 --genmipmap 류의 옵션
        "\"" + pngPath.string() + "\" "
        "\"" + ktxPath.string() + "\"";

    std::cout << "[KTX] " << cmd << std::endl;

    int result = std::system(cmd.c_str());
    if (result != 0) {
        throw std::runtime_error("ktx create failed for: " + pngPath.string());
    }
}

// PBR 관련 키워드가 파일명에 들어있는지 체크
bool TextureManager::IsPbrTextureName(const std::string& name)
{
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return std::tolower(c); });

    for (const char* kw : keywords) {
        if (lower.find(kw) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// 폴더 안의 PBR 관련 PNG -> KTX 자동 변환
void TextureManager::ConvertPbrPngsInFolderToKtx(const std::string& folderPath)
{
    fs::path folder(folderPath);

    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        throw std::runtime_error("Folder does not exist or is not a directory: " + folderPath);
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        fs::path pngPath = entry.path();
        if (pngPath.extension() != ".png") continue;

        const std::string filename = pngPath.filename().string();
        if (!IsPbrTextureName(filename)) {
            continue; // PBR 관련 이름 아니면 무시
        }

        fs::path ktxPath = pngPath;
        ktxPath.replace_extension(".ktx");

        if (fs::exists(ktxPath)) {
            std::cout << "[KTX] exists, skip: " << ktxPath.string() << std::endl;
            continue;
        }

        std::cout << "[KTX] create: " << pngPath.string()
            << " -> " << ktxPath.string() << std::endl;

        CreateKtxFromPng(pngPath, ktxPath);
    }
}