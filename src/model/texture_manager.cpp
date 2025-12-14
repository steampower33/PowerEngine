
#include "texture.h"
#include "context.h"

#include "texture_manager.h"

namespace fs = std::filesystem;

TextureManager::TextureManager(Context& context)
	: context_(context)
{
	{
		ConvertFileToKtx("assets/lut");
		ConvertFileToKtx("assets/SheenCloth");
	}

	{
		vulkan_thumbnail_index_ = CreateTexture("assets", "vulkan_cloth_rgba");

		skybox_index_.morning_env = CreateTexture("assets/DaySky", "env", true);
		skybox_index_.morning_specular = CreateTexture("assets/DaySky", "specular", true);
		skybox_index_.morning_diffuse = CreateTexture("assets/DaySky", "diffuse", true);

		skybox_index_.evening_env = CreateTexture("assets/EveningSky", "env", true);
		skybox_index_.evening_specular = CreateTexture("assets/EveningSky", "specular", true);
		skybox_index_.evening_diffuse = CreateTexture("assets/EveningSky", "diffuse", true);

		skybox_index_.night_env = CreateTexture("assets/NightSky", "env", true);
		skybox_index_.night_specular = CreateTexture("assets/NightSky", "specular", true);
		skybox_index_.night_diffuse = CreateTexture("assets/NightSky", "diffuse", true);
	}

	{
		brdf_index_.ggx = CreateTexture("assets/lut", "ggx", false);
		brdf_index_.charlie = CreateTexture("assets/lut", "charlie", false);
		brdf_index_.sheen_e = CreateTexture("assets/lut", "sheen_e", false);
	}

	{
		keyword_index_[kAlbedoKeyword].push_back(CreateTexture("assets/SheenCloth", "color", false));
		keyword_index_[kNormalKeyword].push_back(CreateTexture("assets/SheenCloth", "normal", false));
		keyword_index_[kSheenKeyword].push_back(CreateTexture("assets/SheenCloth", "sheen", false));
	}

	cnt_tex2d_ = tex2d_.size();

	CreateSetLayouts();
	CreateDescriptorPool();
	CreateDescriptorSets();
}

TextureManager::~TextureManager()
{

}

bool TextureManager::IsRightTextureName(const std::string& name)
{
	std::string lower = name;
	std::transform(lower.begin(), lower.end(), lower.begin(),
		[](unsigned char c) { return std::tolower(c); });

	for (const char* kw : create_keywords_) {
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


int TextureManager::CreateTexture(std::string path, std::string findWord, bool isCubemap)
{
	auto CreateTexture = [&](std::string path, std::string filename)
		{
			std::string texPath = path + "/" + filename;
			std::cout << "Load " << texPath << std::endl;
			fs::path p(texPath);
			std::unique_ptr<Texture> texture = std::make_unique<Texture>(path, filename, context_);
			if (isCubemap)
			{
				uint32_t idx = tex_env_.size();
				tex_env_.push_back(std::move(texture));
				return idx;

			}
			else
			{
				uint32_t idx = tex2d_.size();
				tex2d_.push_back(std::move(texture));
				return idx;
			}
		};

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (!entry.is_regular_file()) continue;

		std::filesystem::path p(entry.path());

		std::string parentPath = p.parent_path().string();
		std::string filename = p.filename().string();

		if (p.extension() != ".ktx" && p.extension() != ".ktx2") continue;

		std::for_each(filename.begin(), filename.end(), [](auto& c) {c = tolower(c); });
		if (filename.find(findWord) != std::string::npos)
		{
			return CreateTexture(parentPath, filename);
		}

	}

	return -1;
}

void TextureManager::CreateSetLayouts()
{
	// Tex2D
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eCombinedImageSampler,
				max_tex2d_,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.sampler += max_tex2d_;
		counts_.layout += 1;

		std::array<vk::DescriptorBindingFlags, 1> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound |
			vk::DescriptorBindingFlagBits::eVariableDescriptorCount
		};

		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
			.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
			.pBindingFlags = bindingFlags.data()
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.pNext = &flagsInfo,
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.tex2d = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// TexEnv
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eCombinedImageSampler,
				max_tex_env_,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.sampler += max_tex_env_;
		counts_.layout += 1;

		std::array<vk::DescriptorBindingFlags, 1> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound |
			vk::DescriptorBindingFlagBits::eVariableDescriptorCount
		};

		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
			.bindingCount = static_cast<uint32_t>(bindingFlags.size()),
			.pBindingFlags = bindingFlags.data()
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.pNext = &flagsInfo,
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.tex_env = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

}

void TextureManager::CreateDescriptorPool()
{
	std::vector<vk::DescriptorPoolSize> poolSizes;

	if (counts_.ubo > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, counts_.ubo);
	}
	if (counts_.ubo_dynamic > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic, counts_.ubo_dynamic);
	}
	if (counts_.sampler > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, counts_.sampler);
	}
	if (counts_.sb > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, counts_.sb);
	}

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = counts_.layout,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	descriptor_pool_ = vk::raii::DescriptorPool(context_.device_, poolInfo);
}

void TextureManager::CreateDescriptorSets()
{
	// tex2D
	{
		vk::DescriptorSetVariableDescriptorCountAllocateInfo countInfo{
			.descriptorSetCount = 1,
			.pDescriptorCounts = &max_tex2d_
		};

		vk::DescriptorSetAllocateInfo allocInfo{
			.pNext = &countInfo,
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.tex2d
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.tex2d = std::move(sets.front());

		// Update
		std::vector<vk::DescriptorImageInfo> imageInfos;
		for (auto& tex : tex2d_) {
			imageInfos.push_back(vk::DescriptorImageInfo{
				.sampler = *tex->texture_sampler_,
				.imageView = *tex->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
				});
		}
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.tex2d,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<uint32_t>(imageInfos.size()),
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = imageInfos.data(),
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// TexEnv
	{
		vk::DescriptorSetVariableDescriptorCountAllocateInfo countInfo{
			.descriptorSetCount = 1,
			.pDescriptorCounts = &max_tex_env_
		};

		vk::DescriptorSetAllocateInfo allocInfo{
			.pNext = &countInfo,
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.tex_env
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.tex_env = std::move(sets.front());

		// Update
		std::vector<vk::DescriptorImageInfo> imageInfos;
		for (auto& tex : tex_env_) {
			imageInfos.push_back(vk::DescriptorImageInfo{
				.sampler = *tex->texture_sampler_,
				.imageView = *tex->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
				});
		}
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.tex_env,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<uint32_t>(imageInfos.size()),
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = imageInfos.data(),
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}
}