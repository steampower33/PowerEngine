#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <cassert>
#include <chrono>
#include <unordered_map>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RIGHT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

inline const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "vertex.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

inline uint32_t findMemoryType(vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

inline void createImage(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
	vk::ImageCreateInfo imageInfo{ .imageType = vk::ImageType::e2D, .format = format,
								  .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
								  .samples = vk::SampleCountFlagBits::e1, .tiling = tiling,
								  .usage = usage, .sharingMode = vk::SharingMode::eExclusive };

	image = vk::raii::Image(device, imageInfo);

	vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size,
									 .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties) };
	imageMemory = vk::raii::DeviceMemory(device, allocInfo);
	image.bindMemory(imageMemory, 0);
}

inline vk::raii::ImageView createImageView(vk::raii::Device& device, vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
	vk::ImageViewCreateInfo viewInfo{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = { aspectFlags, 0, 1, 0, 1 }
	};
	return vk::raii::ImageView(device, viewInfo);
}

inline vk::Format findSupportedFormat(vk::raii::PhysicalDevice& physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
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

inline vk::Format findDepthFormat(vk::raii::PhysicalDevice& physicalDevice) {
	return findSupportedFormat(physicalDevice,
		{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment
	);
}

inline void createBuffer(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
	vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
	buffer = vk::raii::Buffer(device, bufferInfo);
	vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties) };
	bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
	buffer.bindMemory(bufferMemory, 0);
}

inline void copyBuffer(vk::raii::Device& device, vk::raii::CommandPool& commandPool, vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size, vk::raii::Queue& queue) {
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
	vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
	commandCopyBuffer.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
	commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
	commandCopyBuffer.end();
	queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer }, nullptr);
	queue.waitIdle();
}

template<class T>
inline void createVertexBuffer(vk::raii::Device& device,
	vk::raii::PhysicalDevice& physicalDevice,
	vk::raii::CommandPool& commandPool,
	vk::raii::Queue& queue,
	const std::vector<T>& vertices,                 // ★ 인자로 받기
	vk::raii::Buffer& vertexBuffer,
	vk::raii::DeviceMemory& vertexBufferMemory)
{
	vk::DeviceSize bufferSize = sizeof(T) * vertices.size();

	vk::raii::Buffer stagingBuffer(nullptr);
	vk::raii::DeviceMemory stagingMemory(nullptr);
	createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingMemory);

	void* data = stagingMemory.mapMemory(0, bufferSize);
	std::memcpy(data, vertices.data(), (size_t)bufferSize);
	stagingMemory.unmapMemory();

	createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		vertexBuffer, vertexBufferMemory);

	copyBuffer(device, commandPool, stagingBuffer, vertexBuffer, bufferSize, queue);
}

template<class IndexT>
inline void createIndexBuffer(
	vk::raii::Device& device,
	vk::raii::PhysicalDevice& physicalDevice,
	vk::raii::CommandPool& commandPool,
	vk::raii::Queue& queue,
	const std::vector<IndexT>& indices,             // ★ 인자로 받기
	vk::raii::Buffer& indexBuffer,
	vk::raii::DeviceMemory& indexBufferMemory)
{
	vk::DeviceSize bufferSize = sizeof(IndexT) * indices.size();

	vk::raii::Buffer stagingBuffer(nullptr);
	vk::raii::DeviceMemory stagingMemory(nullptr);
	createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingMemory);

	void* data = stagingMemory.mapMemory(0, bufferSize);
	std::memcpy(data, indices.data(), (size_t)bufferSize);
	stagingMemory.unmapMemory();

	createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		indexBuffer, indexBufferMemory);

	copyBuffer(device, commandPool, stagingBuffer, indexBuffer, bufferSize, queue);
}