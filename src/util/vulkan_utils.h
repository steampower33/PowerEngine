#pragma once

namespace vku
{
	enum CpuOrGpu {
		CPU,
		GPU
	};

	enum DepthState {
		MOUSE_DEPTH_NONE = 0,
		MOUSE_DEPTH_IN = 1,
		MOUSE_DEPTH_OUT = 2
	};

	struct Counts {
		uint32_t ubo = 0;
		uint32_t ubo_dynamic = 0;
		uint32_t sb = 0;
		uint32_t sampler = 0;
		uint32_t layout = 0;
	};

	struct TestScene {
		bool sphereCollision = false;
		bool pinnedCorner = false;
		bool topPinnedCorner = false;
		bool selfCollision = false;
	};

	struct VertexIncludeInfo {
		bool uv = false;
		bool normal = false;
		bool tangent = false;
	};

	enum PolygonMode {
		SOLID,
		WIREFRAME,
		POINT
	};

	inline vk::SampleCountFlagBits GetMaxUsableSampleCount(vk::PhysicalDeviceProperties physicalDeviceProperties) {
		vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
		if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
		if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
		if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
		if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
		if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

		return vk::SampleCountFlagBits::e1;
	}

	inline void TransitionImageLayout(
		vk::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	) {
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = src_stage_mask,
			.srcAccessMask = src_access_mask,
			.dstStageMask = dst_stage_mask,
			.dstAccessMask = dst_access_mask,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};
		vk::DependencyInfo dependency_info = {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier
		};
		cmd.pipelineBarrier2(dependency_info);
	}

	inline void TransitionImageLayoutCustom(
		vk::raii::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags aspect_mask
	)
	{
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = src_stage_mask,
			.srcAccessMask = src_access_mask,
			.dstStageMask = dst_stage_mask,
			.dstAccessMask = dst_access_mask,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = *image,
			.subresourceRange = {
				.aspectMask = aspect_mask,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};
		vk::DependencyInfo dependency_info = {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier
		};
		cmd.pipelineBarrier2(dependency_info);
	}

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

	inline void CreateImage(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		uint32_t                   width,
		uint32_t                   height,
		uint32_t                   mipLevels,
		vk::SampleCountFlagBits    numSamples,
		vk::Format                 format,
		vk::ImageTiling            tiling,
		vk::ImageUsageFlags        usage,
		vk::MemoryPropertyFlags    properties,
		vk::raii::Image& image,
		vk::raii::DeviceMemory& imageMemory,
		vk::ImageCreateFlags       flags = {},
		uint32_t                   arrayLayers = 1,
		vk::ImageType              imageType = vk::ImageType::e2D
	) {
		vk::ImageCreateInfo imageInfo{
		.flags = flags,
		.imageType = imageType,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = mipLevels,
		.arrayLayers = arrayLayers,
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
			.memoryTypeIndex = FindMemoryType(
				physicalDevice,
				memRequirements.memoryTypeBits,
				properties)
		};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);
	}

	inline vk::raii::ImageView CreateImageView(vk::raii::Device& device, vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels, vk::ImageViewType viewType = vk::ImageViewType::e2D, uint32_t faces = 1u) {
		vk::ImageViewCreateInfo viewInfo{
				.image = image,
				.viewType = viewType,
				.format = format,
				.subresourceRange = { aspectFlags, 0, mipLevels, 0, faces }
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

	inline void CreateBuffer(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device,  vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) 
	{
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

	template<typename T>
	inline void CreateVertexBuffer(
		vk::raii::PhysicalDevice& physicalDevice, 
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		const std::vector<T>& vertices,
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

	template<typename T>
	inline void CreateIndexBuffer(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		const std::vector<T>& indices,
		vk::raii::Buffer& indexBuffer,
		vk::raii::DeviceMemory& indexBufferMemory)
	{
		vk::DeviceSize bufferSize = sizeof(T) * indices.size();

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

	template <typename T>
	inline void CreateSSBO(
		vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Device& device,
		vk::raii::Queue& queue,
		vk::raii::CommandPool& commandPool,
		vk::DeviceSize bufferSize,
		vk::BufferUsageFlags stagingUsage, 
		vk::MemoryPropertyFlags stagingProperties,
		std::vector<T>& data,
		vk::BufferUsageFlags ssboUsage,
		vk::MemoryPropertyFlags ssboProperties,
		vk::raii::Buffer& ssbo, 
		vk::raii::DeviceMemory& ssboMem,
		vk::raii::Buffer* optStagingBuf = nullptr,
		vk::raii::DeviceMemory* optStagingMem = nullptr
	)
	{
		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});

		if (optStagingBuf && optStagingMem) {
			stagingBuffer = std::move(*optStagingBuf);
			stagingBufferMemory = std::move(*optStagingMem);
			vku::CreateBuffer(physicalDevice, device, bufferSize, stagingUsage, stagingProperties, stagingBuffer, stagingBufferMemory);
		}
		else {
			vku::CreateBuffer(physicalDevice, device, bufferSize,
				stagingUsage, stagingProperties,
				stagingBuffer, stagingBufferMemory);
		}

		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, data.data(), (size_t)bufferSize);
		stagingBufferMemory.unmapMemory();

		ssbo.clear();
		ssboMem.clear();

		vk::raii::Buffer shaderStorageBufferTemp({});
		vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
		vku::CreateBuffer(physicalDevice, device, bufferSize, ssboUsage, ssboProperties, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
		vku::CopyBuffer(device, queue, commandPool, stagingBuffer, shaderStorageBufferTemp, bufferSize);
		ssbo = std::move(shaderStorageBufferTemp);
		ssboMem = std::move(shaderStorageBufferTempMemory);

		if (optStagingBuf && optStagingMem) {
			*optStagingBuf = std::move(stagingBuffer);
			*optStagingMem = std::move(stagingBufferMemory);
		}
	}

	inline void BufferBarrier2(
		const vk::raii::CommandBuffer& cmd,
		const vk::Buffer& buffer,
		vk::DeviceSize offset,
		vk::DeviceSize size,
		vk::PipelineStageFlags2 srcStage,
		vk::AccessFlags2        srcAccess,
		vk::PipelineStageFlags2 dstStage,
		vk::AccessFlags2        dstAccess)
	{
		vk::BufferMemoryBarrier2 buf{};
		buf.srcStageMask = srcStage;
		buf.srcAccessMask = srcAccess;
		buf.dstStageMask = dstStage;
		buf.dstAccessMask = dstAccess;
		buf.buffer = buffer;
		buf.offset = offset;
		buf.size = size;

		vk::DependencyInfo dep{};
		dep.setBufferMemoryBarriers(buf);

		cmd.pipelineBarrier2(dep);
	}

	inline void ssboCompWtoCompR(
		const vk::raii::CommandBuffer& cmd,
		const vk::Buffer& buffer)
	{
		BufferBarrier2(
			cmd, buffer, 0, VK_WHOLE_SIZE,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead
		);
	}

	inline void ssboCompWtoCompW(
		const vk::raii::CommandBuffer& cmd,
		const vk::Buffer& buffer)
	{
		BufferBarrier2(
			cmd, buffer, 0, VK_WHOLE_SIZE,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite
		);
	}

	inline void ssboCompWtoCompRW(
		const vk::raii::CommandBuffer& cmd,
		const vk::Buffer& buffer)
	{
		BufferBarrier2(
			cmd, buffer, 0, VK_WHOLE_SIZE,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite
		);
	}

	inline void ssboCompWtoVertR(
		const vk::raii::CommandBuffer& cmd,
		const vk::Buffer& buffer)
	{
		BufferBarrier2(
			cmd, buffer, 0, VK_WHOLE_SIZE,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eVertexShader,
			vk::AccessFlagBits2::eShaderStorageRead
		);
	}

	inline void Barrier2(
		const vk::raii::CommandBuffer& cmd,
		vk::PipelineStageFlags2 srcStage,
		vk::AccessFlags2        srcAccess,
		vk::PipelineStageFlags2 dstStage,
		vk::AccessFlags2        dstAccess)
	{
		vk::MemoryBarrier2 mem{};
		mem.srcStageMask = srcStage;
		mem.srcAccessMask = srcAccess;
		mem.dstStageMask = dstStage;
		mem.dstAccessMask = dstAccess;

		std::array<vk::MemoryBarrier2, 1> mems{ mem };

		vk::DependencyInfo dep{};
		dep.setMemoryBarriers(mems);

		cmd.pipelineBarrier2(dep);
	}

	template <typename T>
	inline void CopyStagingToSSBO(const vk::raii::CommandBuffer& cmd, VkDeviceSize size, void* map, std::vector<T>& data, vk::raii::Buffer& staging, vk::raii::Buffer& ssbo, vk::PipelineStageFlagBits2 srcStageMask, vk::AccessFlagBits2 srcAccessMask, vk::PipelineStageFlagBits2 dstStageMask, vk::AccessFlagBits2 dstAccessMask)
	{
		// 1) staging memcpy
		std::memcpy(map, data.data(), (size_t)size);
		
		// 2) copy staging -> device
		vk::BufferCopy region{ 0, 0, size };
		cmd.copyBuffer(*staging, *ssbo, { region });

		// 3) barrier: TRANSFER_WRITE -> VERTEX/SSBO read
		vk::BufferMemoryBarrier2 b{
			.srcStageMask = srcStageMask,
			.srcAccessMask = srcAccessMask,
			.dstStageMask = dstStageMask,
			.dstAccessMask = dstAccessMask,
			.buffer = *ssbo,
			.offset = 0,
			.size = size
		};
		vk::DependencyInfo dep{
			.bufferMemoryBarrierCount = 1,
			.pBufferMemoryBarriers = &b
		};
		cmd.pipelineBarrier2(dep);

	}

}