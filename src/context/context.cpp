
#include "context.h"

Context::Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height)
	: glfw_window_(glfwWindow)
{
	CreateInstance();
	SetupDebugMessenger();
	CreateSurface();
	PickPhysicalDevice();
	CreateLogicalDevice();
	CreateCommandPool();
}


Context::~Context()
{
}

void Context::CreateInstance() {
	constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Power Engine",
				.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
				.pEngineName = "Power Engine",
				.engineVersion = VK_MAKE_VERSION(1, 0, 0),
				.apiVersion = vk::ApiVersion14 };

	// Get the required layers
	std::vector<char const*> requiredLayers;
	if (enableValidationLayers) {
		requiredLayers.assign(validation_layers.begin(), validation_layers.end());
	}

	vk::raii::Context context;
	// Check if the required layers are supported by the Vulkan implementation.
	auto layerProperties = context.enumerateInstanceLayerProperties();
	for (auto const& requiredLayer : requiredLayers)
	{
		if (std::ranges::none_of(layerProperties,
			[requiredLayer](auto const& layerProperty)
			{ return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
		{
			throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
		}
	}

	// Get the required extensions.
	auto requiredExtensions = GetRequiredExtensions();

	// Check if the required extensions are supported by the Vulkan implementation.
	auto extensionProperties = context.enumerateInstanceExtensionProperties();
	for (auto const& requiredExtension : requiredExtensions)
	{
		if (std::ranges::none_of(extensionProperties,
			[requiredExtension](auto const& extensionProperty)
			{ return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
		{
			throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
		}
	}

	vk::InstanceCreateInfo createInfo{
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
		.ppEnabledLayerNames = requiredLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data() };
	instance_ = vk::raii::Instance(context, createInfo);
}

std::vector<const char*> Context::GetRequiredExtensions() {
	uint32_t glfwExtensionCount = 0;
	auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	if (enableValidationLayers) {
		extensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	return extensions;
}

void Context::SetupDebugMessenger() {
	if (!enableValidationLayers) return;

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags,
		.messageType = messageTypeFlags,
		.pfnUserCallback = &DebugCallback
	};
	debug_messenger_ = instance_.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL Context::DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
	if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
		std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
	}

	return vk::False;
}

void Context::CreateSurface() {
	VkSurfaceKHR       _surface;
	if (glfwCreateWindowSurface(*instance_, glfw_window_, nullptr, &_surface) != 0) {
		throw std::runtime_error("failed to create window surface!");
	}
	surface_ = vk::raii::SurfaceKHR(instance_, _surface);

}

void Context::PickPhysicalDevice() {
	std::vector<vk::raii::PhysicalDevice> devices = instance_.enumeratePhysicalDevices();
	const auto                            devIter = std::ranges::find_if(
		devices,
		[&](auto const& device)
		{
			// Check if the device supports the Vulkan 1.3 API version
			bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

			// Check if any of the queue families support graphics operations
			auto queueFamilies = device.getQueueFamilyProperties();
			bool supportsGraphics =
				std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

			// Check if all required device extensions are available
			auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
			auto hasAtomicFloatExt = std::any_of(
				availableDeviceExtensions.begin(), availableDeviceExtensions.end(),
				[](const vk::ExtensionProperties& e) {
					return std::strcmp(e.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0;
				});

			bool supportsAllRequiredExtensions =
				std::ranges::all_of(required_device_extension_,
					[&availableDeviceExtensions](auto const& requiredDeviceExtension)
					{
						return std::ranges::any_of(availableDeviceExtensions,
							[requiredDeviceExtension](auto const& availableDeviceExtension)
							{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
					});
			auto features = device.template getFeatures2<
				vk::PhysicalDeviceFeatures2,
				vk::PhysicalDeviceVulkan13Features,
				vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
				vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
				vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT,
				vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>();

			const auto& coreFeats = features.template get<vk::PhysicalDeviceFeatures2>().features;
			const auto& v13Feats = features.template get<vk::PhysicalDeviceVulkan13Features>();
			const auto& dynState = features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
			const auto& timeline = features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
			const auto& indexing = features.template get<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>();

			bool supportsRequiredFeatures =
				coreFeats.samplerAnisotropy &&
				v13Feats.dynamicRendering &&
				dynState.extendedDynamicState &&
				timeline.timelineSemaphore &&

				// 여기부터 descriptor indexing 관련
				indexing.runtimeDescriptorArray &&
				indexing.shaderSampledImageArrayNonUniformIndexing &&
				indexing.descriptorBindingPartiallyBound &&
				indexing.descriptorBindingVariableDescriptorCount;

			return supportsRequiredFeatures;
		});
	if (devIter != devices.end())
	{
		physical_device_ = *devIter;
	}
	else
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}


void Context::CreateLogicalDevice() {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physical_device_.getQueueFamilyProperties();

	// get the first index into queueFamilyProperties which supports both graphics and present
	for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
	{
		if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			(queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
			physical_device_.getSurfaceSupportKHR(qfpIndex, *surface_))
		{
			// found a queue family that supports both graphics and present
			queue_index_ = qfpIndex;
			break;
		}
	}
	if (queue_index_ == ~0)
	{
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
	}

	vk::StructureChain<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
		vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT,
		vk::PhysicalDeviceDescriptorIndexingFeaturesEXT> featureChain = {
		// 0: core features2
		{
			.features = {
				.sampleRateShading = vk::True,
				.fillModeNonSolid = vk::True,
				.samplerAnisotropy = vk::True,
			},
		},
		// 1: Vulkan 1.3
		{
			.synchronization2 = vk::True,
			.dynamicRendering = vk::True
		},
		// 2: Extended dynamic state
		{
			.extendedDynamicState = vk::True
		},
		// 3: Timeline semaphore
		{
			.timelineSemaphore = vk::True
		},
		// 4: Atomic float
		{
			.shaderBufferFloat32Atomics = vk::True,
			.shaderBufferFloat32AtomicAdd = vk::True
		},
		// 5: Descriptor indexing
		{
			.shaderSampledImageArrayNonUniformIndexing = vk::True,
			.descriptorBindingPartiallyBound = vk::True,
			.descriptorBindingVariableDescriptorCount = vk::True,
			.runtimeDescriptorArray = vk::True,
		}
	};

	// create a Device
	float                     queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queue_index_, .queueCount = 1, .pQueuePriorities = &queuePriority };
	vk::DeviceCreateInfo      deviceCreateInfo{
		.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<uint32_t>(required_device_extension_.size()),
		.ppEnabledExtensionNames = required_device_extension_.data()
	};

	device_ = vk::raii::Device(physical_device_, deviceCreateInfo);
	queue_ = vk::raii::Queue(device_, queue_index_, 0);
}


void Context::CreateCommandPool() {
	vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
										 .queueFamilyIndex = queue_index_ };
	command_pool_ = vk::raii::CommandPool(device_, poolInfo);
}

void Context::WaitIdle()
{
	device_.waitIdle();
}