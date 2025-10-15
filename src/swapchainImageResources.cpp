#include "swapchainImageResources.hpp"

SwapchainImageResources::SwapchainImageResources(vk::Image& image, vk::SurfaceFormatKHR& swapchainSurfaceFormat, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::Extent2D& swapchainExtent) :
	image_(image)
{
	createSwapchainImageView(device, swapchainSurfaceFormat.format);

	vk::Format depthFormat = findDepthFormat(physicalDevice);
	createImage(device, physicalDevice, swapchainExtent.width, swapchainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage_, depthMem_);
	depthImageView_ = createImageView(device, depthImage_, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

SwapchainImageResources::~SwapchainImageResources()
{

}

void SwapchainImageResources::createSwapchainImageView(vk::raii::Device& device, vk::Format format)
{
	vk::ImageViewCreateInfo imageViewCreateInfo{ 
		.image = image_,
		.viewType = vk::ImageViewType::e2D, 
		.format = format,
	  .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
	};
	imageView_ = vk::raii::ImageView{ device, imageViewCreateInfo };
}