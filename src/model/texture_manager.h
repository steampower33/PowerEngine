#pragma once

class Texture;

class TextureManager
{
public:
	TextureManager(Context& context);
	TextureManager(const TextureManager& rhs) = delete;
	TextureManager(TextureManager&& rhs) = delete;
	TextureManager& operator=(const TextureManager& rhs) = delete;
	TextureManager& operator=(TextureManager&& rhs) = delete;
	~TextureManager();

	Context& context_;

	const uint32_t max_texture_size = 32;
	std::vector<std::unique_ptr<Texture>> textures_;

	const uint32_t env_texture_size = 3;
	std::vector<std::unique_ptr<Texture>> env_textures_;
	
	uint32_t vulkan_thumbnail_index_ = 0;
	uint32_t brdf_lut_index_ = 0;

	const char* keywords_[10] = {
		"color",
		"albedo",
		"metallic",
		"metalness",
		"normal",
		"roughness",
		"ao",
		"height",
		"displacement",
		"lut",
	};

	void ConvertFileToKtx(const std::string& folderPath);
	bool IsRightTextureName(const std::string& name);
	void CreateKtxFromFile(const std::filesystem::path& pngPath, const std::filesystem::path& ktxPath);
	uint32_t CreateTexture(std::string path, std::string keyword, bool isCubemap = false);
};