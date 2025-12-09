#pragma once

class Texture;
class Context;

#include "vulkan_utils.h"

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

	vku::Count counts_;

	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	struct SetLayout {
		vk::raii::DescriptorSetLayout tex2d{ nullptr };
		vk::raii::DescriptorSetLayout tex_env{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet tex2d{ nullptr };
		vk::raii::DescriptorSet tex_env{ nullptr };
	} sets_;

	const uint32_t max_tex2d_ = 32;
	std::vector<std::unique_ptr<Texture>> tex2d_;

	const uint32_t max_tex_env_ = 32;
	std::vector<std::unique_ptr<Texture>> tex_env_;

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

	struct BrdfIndex {
		int ggx = -1;
		int charlie = -1;
		int sheen_e = -1;
	} brdf_index_;

	struct SkyboxIndex {
		int morning_env = -1;
		int morning_specular = -1;
		int morning_diffuse = -1;
		int evening_env = -1;
		int evening_specular = -1;
		int evening_diffuse = -1;
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

private:
	void CreateSetLayouts();
	void CreateDescriptorPool();
	void CreateDescriptorSets();
};