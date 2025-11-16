#pragma once

class Texture2D;

class TextureManager
{
public:
	TextureManager(Context& context, GraphicsContext& graphicsContext);
	TextureManager(const TextureManager& rhs) = delete;
	TextureManager(TextureManager&& rhs) = delete;
	TextureManager& operator=(const TextureManager& rhs) = delete;
	TextureManager& operator=(TextureManager&& rhs) = delete;
	~TextureManager();

	std::unique_ptr<Texture2D> vulkan_title_image_{ nullptr };

	std::unique_ptr<Texture2D> worm_albedo_{ nullptr };
	std::unique_ptr<Texture2D> worm_ao_{ nullptr };
	std::unique_ptr<Texture2D> worm_roughness_{ nullptr };
	std::unique_ptr<Texture2D> worm_metallic_{ nullptr };
	std::unique_ptr<Texture2D> worm_height_{ nullptr };
	std::unique_ptr<Texture2D> worm_normal_{ nullptr };

	const char* keywords[6] = {
		"albedo",
		"ao",
		"roughness",
		"metallic",
		"height",
		"normal",
	};

	void ConvertPbrPngsInFolderToKtx(const std::string& folderPath);
	bool IsPbrTextureName(const std::string& name);
	void CreateKtxFromPng(const std::filesystem::path& pngPath, const std::filesystem::path& ktxPath);
};