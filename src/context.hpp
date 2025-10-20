#pragma once

#include "swapchain.hpp"

struct Camera;

class Context
{
public:
	Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	~Context();

	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;

	void draw(Camera& camera, float dt);

	vk::raii::Context                context_;
	vk::raii::Instance               instance_ = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
	vk::raii::SurfaceKHR             surface_ = nullptr;
	vk::raii::PhysicalDevice         physicalDevice_ = nullptr;
	vk::SampleCountFlagBits			 msaaSamples_ = vk::SampleCountFlagBits::e1;
	vk::raii::Device                 device_ = nullptr;
	uint32_t                         queueIndex_ = ~0;
	vk::raii::Queue                  queue_ = nullptr;
	vk::raii::CommandPool			 commandPool_ = nullptr;
	vk::raii::DescriptorPool		 descriptorPool_ = nullptr;

	uint32_t					 	 uboCount_ = 0;
	uint32_t					 	 sbCount_ = 0;
	uint32_t						 samplerCount_ = 0;
	uint32_t						 layoutCount_ = 0;

	DescriptorSetLayouts			 descriptorSetLayouts_;
	Pipelines						 pipelines_;

	bool framebufferResized_ = false;

	std::vector<const char*> requiredDeviceExtension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName
	};

	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

	// ImGUI
	vk::raii::DescriptorPool imguiPool_ = nullptr;
	void setupImgui(uint32_t width, uint32_t height);
	void drawImgui();

private:
	GLFWwindow* glfwWindow_;
	std::unique_ptr<Swapchain> swapchain_;

	void createInstance();
	std::vector<const char*> getRequiredExtensions();
	void setupDebugMessenger();
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);
	void createSurface();
	void pickPhysicalDevice();
	vk::SampleCountFlagBits getMaxUsableSampleCount();
	void createLogicalDevice();

	void createModelDescriptorSetLayout();
	void createParticleDescriptorSetLayout();
	void createDescriptorPool();
	
	void createModelPipeline();

	void createParticleComputePipeline();
	void createParticleGraphicsPipeline();

	void createCommandPool();

	static std::vector<char> readFile(const std::string& filename);
};