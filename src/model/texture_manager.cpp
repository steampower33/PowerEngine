
#include "texture.h"

#include "texture_manager.h"

namespace fs = std::filesystem;

TextureManager::TextureManager(Context& context)
    : context_(context)
{
    ConvertFileToKtx("assets/Metal");
    ConvertFileToKtx("assets/lut");
    ConvertFileToKtx("assets/Fabric");

    vulkan_thumbnail_index_ = CreateTexture("assets", "vulkan_cloth_rgba");
    brdf_lut_index_ = CreateTexture("assets/lut", "lut_ggx");
}

TextureManager::~TextureManager()
{

}

// PBR 관련 키워드가 파일명에 들어있는지 체크
bool TextureManager::IsRightTextureName(const std::string& name)
{
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return std::tolower(c); });

    for (const char* kw : keywords_) {
        if (lower.find(kw) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// png, jpg -> ktx
void TextureManager::ConvertFileToKtx(const std::string& folderPath)
{
    fs::path folder(folderPath);

    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        throw std::runtime_error("Folder does not exist or is not a directory: " + folderPath);
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        fs::path pngPath = entry.path();
        if (pngPath.extension() != ".png" && 
            pngPath.extension() != ".jpg")
            continue;

        const std::string filename = pngPath.filename().string();
        if (!IsRightTextureName(filename)) {
            continue;
        }

        fs::path ktxPath = pngPath;
        ktxPath.replace_extension(".ktx2");

        if (fs::exists(ktxPath)) {
            std::cout << "[KTX] exists, skip: " << ktxPath.string() << std::endl;
            continue;
        }

        std::cout << "[KTX] create: " << pngPath.string()
            << " -> " << ktxPath.string() << std::endl;

        CreateKtxFromFile(pngPath, ktxPath);
    }
}

void TextureManager::CreateKtxFromFile(const fs::path& pngPath, const fs::path& ktxPath)
{
    std::string filename = pngPath.filename().string();
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });

    bool isAlbedo =
        lower.find("color") != std::string::npos ||
        lower.find("albedo") != std::string::npos ||
        lower.find("basecolor") != std::string::npos ||
        lower.find("base_color") != std::string::npos;

    bool isLut =
        lower.find("lut") != std::string::npos;

    std::string fmt;
    std::string tf;
    bool genMipmap = true;

    if (isAlbedo) {
        fmt = "R8G8B8A8_SRGB";
        tf = "srgb";
    }
    else if (isLut) {
        fmt = "R8G8B8A8_UNORM";
        tf = "linear";
        genMipmap = false;
    }
    else {
        fmt = "R8G8B8A8_UNORM";
        tf = "linear";
    }

    std::string cmd = "ktx create "
        "--format " + fmt + " "
        "--assign-tf " + tf + " ";

    if (!isLut) {
        cmd += "--assign-primaries bt709 ";
    }

    if (genMipmap) {
        cmd += "--generate-mipmap ";
    }

    cmd += "\"" + pngPath.string() + "\" "
        "\"" + ktxPath.string() + "\"";

    std::cout << "[KTX] " << cmd << std::endl;

    int result = std::system(cmd.c_str());
    if (result != 0) {
        throw std::runtime_error("ktx2 create failed for: " + pngPath.string());
    }
}


uint32_t TextureManager::CreateTexture(std::string path, std::string keyword, bool isCubemap)
{
    auto CreateTexture = [&](std::string path, std::string filename)
        {
            std::string texPath = path + "/" + filename;
            std::cout << "Load " << texPath << std::endl;
            fs::path p(texPath);
            std::unique_ptr<Texture> texture = std::make_unique<Texture>(path, filename, context_);
            if (isCubemap)
            {
                uint32_t idx = env_textures_.size();
                env_textures_.push_back(std::move(texture));
                return idx;

            }
            else
            {
                uint32_t idx = textures_.size();
                textures_.push_back(std::move(texture));
                return idx;
            }
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

        if (p.extension() != ".ktx" && p.extension() != ".ktx2") continue;

        std::for_each(filename.begin(), filename.end(), [](auto& c) {c = tolower(c); });
        if (filename.find(keyword) != std::string::npos)
            return CreateTexture(parentPath, filename);

    }

    return 0;
}