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

	const uint32_t env_texture_size = 9;
	std::vector<std::unique_ptr<Texture>> env_textures_;
	
	int vulkan_thumbnail_index_ = 0;

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

	struct SkyboxIndex {
		int morning_brdf = -1;
		int morning_env = -1;
		int morning_specular = -1;
		int morning_diffuse = -1;
		int evening_brdf = -1;
		int evening_env = -1;
		int evening_specular = -1;
		int evening_diffuse = -1;
		int night_brdf = -1;
		int night_env = -1;
		int night_specular = -1;
		int night_diffuse = -1;
	} skybox_index_;

	struct SkyboxEnable {
		bool morning = true;
		bool evening = false;
		bool night = false;
	} skybox_enable_;

	void ConvertFileToKtx(const std::string& folderPath);
	bool IsRightTextureName(const std::string& name);
	void CreateKtxFromFile(const std::filesystem::path& pngPath, const std::filesystem::path& ktxPath);
	int CreateTexture(std::string path, std::string keyword, bool isCubemap = false);
};