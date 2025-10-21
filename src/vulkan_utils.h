#pragma once

namespace vku
{
	inline [[nodiscard]] vk::raii::ShaderModule CreateShaderModule(vk::raii::Device& device, const std::vector<char>& code) {
		vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
		vk::raii::ShaderModule shaderModule{ device, createInfo };

		return shaderModule;
	}

	inline std::vector<char> ReadFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();
		return buffer;
	}

	inline uint32_t FindMemoryType(vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	inline void CreateImage(vk::raii::PhysicalDevice& physicalDevice,  vk::raii::Device& device, uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
		vk::ImageCreateInfo imageInfo{
			   .imageType = vk::ImageType::e2D,
			   .format = format,
			   .extent = {width, height, 1},
			   .mipLevels = mipLevels,
			   .arrayLayers = 1,
			   .samples = numSamples,
			   .tiling = tiling,
			   .usage = usage,
			   .sharingMode = vk::SharingMode::eExclusive,
			   .initialLayout = vk::ImageLayout::eUndefined
		};

		image = vk::raii::Image(device, imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties)
		};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);
	}

	inline vk::raii::ImageView CreateImageView(vk::raii::Device& device, vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
		vk::ImageViewCreateInfo viewInfo{
				.image = image,
				.viewType = vk::ImageViewType::e2D,
				.format = format,
				.subresourceRange = { aspectFlags, 0, mipLevels, 0, 1 }
		};
		return vk::raii::ImageView(device, viewInfo);
	}

	inline vk::Format FindSupportedFormat(vk::raii::PhysicalDevice& physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
		auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);
			return (((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
				((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features)));
			});
		if (formatIt == candidates.end())
		{
			throw std::runtime_error("failed to find supported format!");
		}
		return *formatIt;
	}

	inline vk::Format FindDepthFormat(vk::raii::PhysicalDevice& physicalDevice) {
		return FindSupportedFormat(physicalDevice,
			{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment
		);
	}

	inline void CreateBuffer(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device,  vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
		vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
		buffer = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size, .memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties) };
		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(bufferMemory, 0);
	}

	inline void CopyBuffer(vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
		vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
		vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		commandCopyBuffer.end();
		queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
		queue.waitIdle();
	}

	template<class T>
	inline void CreateVertexBuffer(
		vk::raii::PhysicalDevice& physicalDevice, 
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		const std::vector<T>& vertices,                 // ★ 인자로 받기
		vk::raii::Buffer& vertexBuffer,
		vk::raii::DeviceMemory& vertexBufferMemory)
	{
		vk::DeviceSize bufferSize = sizeof(T) * vertices.size();

		vk::raii::Buffer stagingBuffer(nullptr);
		vk::raii::DeviceMemory stagingMemory(nullptr);
		CreateBuffer(physicalDevice, device, bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingMemory);

		void* data = stagingMemory.mapMemory(0, bufferSize);
		std::memcpy(data, vertices.data(), (size_t)bufferSize);
		stagingMemory.unmapMemory();

		CreateBuffer(physicalDevice, device, bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vertexBuffer, vertexBufferMemory);

		CopyBuffer(device, queue, commandPool, stagingBuffer, vertexBuffer, bufferSize);
	}

	template<class IndexT>
	inline void CreateIndexBuffer(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		const std::vector<IndexT>& indices,             // ★ 인자로 받기
		vk::raii::Buffer& indexBuffer,
		vk::raii::DeviceMemory& indexBufferMemory)
	{
		vk::DeviceSize bufferSize = sizeof(IndexT) * indices.size();

		vk::raii::Buffer stagingBuffer(nullptr);
		vk::raii::DeviceMemory stagingMemory(nullptr);
		CreateBuffer(physicalDevice, device, bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingMemory);

		void* data = stagingMemory.mapMemory(0, bufferSize);
		std::memcpy(data, indices.data(), (size_t)bufferSize);
		stagingMemory.unmapMemory();

		CreateBuffer(physicalDevice, device, bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			indexBuffer, indexBufferMemory);

		CopyBuffer(device, queue, commandPool, stagingBuffer, indexBuffer, bufferSize);
	}
}