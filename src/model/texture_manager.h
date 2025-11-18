#pragma once

class Texture2D;

class TextureManager
{
public:
	TextureManager(Context& context);
	TextureManager(const TextureManager& rhs) = delete;
	TextureManager(TextureManager&& rhs) = delete;
	TextureManager& operator=(const TextureManager& rhs) = delete;
	TextureManager& operator=(TextureManager&& rhs) = delete;
	~TextureManager();

	std::vector<std::unique_ptr<Texture2D>> textures_;
	Context& context_;

	const uint32_t max_texture_size = 32;

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
	uint32_t CreateTexture2D(std::string path, std::string keyword);
};