
#include "texture_2d.h"

#include "texture_manager.h"

namespace fs = std::filesystem;

TextureManager::TextureManager(Context& context, GraphicsContext& graphicsContext)
{
	vulkan_title_image_ = std::make_unique<Texture2D>("assets/vulkan_cloth_rgba.ktx", context, graphicsContext);

    std::string folder = "assets/worm";
    
    ConvertPbrPngsInFolderToKtx(folder);

    auto setTexture = [&](std::string filename, std::unique_ptr<Texture2D>& texture)
        {
            std::string texPath = folder + "/" + filename;
            std::cout << "Load " << texPath << std::endl;
            fs::path p(texPath);
            p.replace_extension(".ktx");
            texture = std::make_unique<Texture2D>(p.string(), context, graphicsContext);
        };

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();

        if (filename.find("ktx") == std::string::npos) continue;

        if (filename.find("albedo") != std::string::npos)
            setTexture(filename, worm_albedo_);
        if (filename.find("ao") != std::string::npos)
            setTexture(filename, worm_ao_);
        if (filename.find("roughness") != std::string::npos)
            setTexture(filename, worm_roughness_);
        if (filename.find("metallic") != std::string::npos)
            setTexture(filename, worm_metallic_);
        if (filename.find("height") != std::string::npos)
            setTexture(filename, worm_height_);
        if (filename.find("normal") != std::string::npos)
            setTexture(filename, worm_normal_);


    }

}

TextureManager::~TextureManager()
{

}


// 개별 PNG를 KTX로 변환
void TextureManager::CreateKtxFromPng(const fs::path& pngPath, const fs::path& ktxPath)
{
    std::string cmd = "ktx create "
        "--format R8G8B8A8_UNORM "
        "--assign-tf linear "
        "--assign-primaries bt709 "
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