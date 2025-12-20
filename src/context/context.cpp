
#include "context.h"

Context::Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height)
	: glfw_window_(glfwWindow)
{
	CreateInstance();
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

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
		createInfo.ppEnabledLayerNames = validation_layers.data();
	}

	// --- Validation features (sync validation, best practices, etc.) ---
	vk::ValidationFeaturesEXT vfe{};
	std::vector<vk::ValidationFeatureEnableEXT> enables;
	vk::DebugUtilsMessengerCreateInfoEXT dbgCreateInfo{};

	if (enableValidationLayers)
	{
		enables = {
			vk::ValidationFeatureEnableEXT::eSynchronizationValidation
		};

		dbgCreateInfo = vk::DebugUtilsMessengerCreateInfoEXT{
			.messageSeverity =
				vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			.messageType =
				vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
				vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
		};

		if (debug_mode_ == DebugMode::NORMAL)
		{
			dbgCreateInfo.pfnUserCallback = &DebugCallback_ErrorOnly;
		}
		else if (debug_mode_ == DebugMode::FOCUS)
		{
			dbgCreateInfo.pfnUserCallback = &DebugCallback_Focus;
			dbgCreateInfo.messageSeverity |=
				vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
				vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
			dbgCreateInfo.messageType |=
				vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

			enables = {
				vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
				vk::ValidationFeatureEnableEXT::eBestPractices,
				vk::ValidationFeatureEnableEXT::eGpuAssisted,
				//vk::ValidationFeatureEnableEXT::eGpuAssistedReserveBindingSlot // optional (descriptor indexing helping
			};
		}
	}

	vfe.setEnabledValidationFeatures(enables);

	// Chain: InstanceCreateInfo -> ValidationFeatures -> DebugCreateInfo
	vk::StructureChain<
		vk::InstanceCreateInfo,
		vk::ValidationFeaturesEXT,
		vk::DebugUtilsMessengerCreateInfoEXT
	> chain{ createInfo, vfe, dbgCreateInfo };

	if (!enableValidationLayers)
	{
		// If validation is off, don't rely on the chain extras
		instance_ = vk::raii::Instance(context, createInfo);
	}
	else
	{
		instance_ = vk::raii::Instance(context, chain.get<vk::InstanceCreateInfo>());
	}
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

VKAPI_ATTR VkBool32 VKAPI_CALL Context::DebugCallback_ErrorOnly(
	vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	vk::DebugUtilsMessageTypeFlagsEXT type,
	const vk::DebugUtilsMessengerCallbackDataEXT* cb,
	void* userData)
{
	(void)type; (void)userData;

	if (!(severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)) {
		return VK_FALSE;
	}

	const char* msg = (cb && cb->pMessage) ? cb->pMessage : "(null)";
	std::cerr << "[Vulkan][ERROR] " << msg << "\n";
	return VK_FALSE;
}

static bool ContainsAny(std::string_view s, std::initializer_list<std::string_view> keys) {
	for (auto k : keys) {
		if (!k.empty() && s.find(k) != std::string_view::npos) return true;
	}
	return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Context::DebugCallback_Focus(
	vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
	vk::DebugUtilsMessageTypeFlagsEXT type,
	const vk::DebugUtilsMessengerCallbackDataEXT* cb,
	void* userData)
{
	(void)type; (void)userData;

	const char* cmsg = (cb && cb->pMessage) ? cb->pMessage : "(null)";
	std::string_view msg(cmsg);

	auto isError = (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	auto isWarning = (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);

	// Errors: always print
	// Warnings: print only if message looks like synchronization-related
	if (isWarning && !isError) {
		// Common sync-validation fingerprints
		const bool looksSync =
			ContainsAny(msg, {
				"hazard",
				"SYNC-",
				"synchronization",
				"WRITE_AFTER_READ",
				"WRITE_AFTER_WRITE",
				"READ_AFTER_WRITE",
				"VkSemaphoreSubmitInfo",
				"pipelineBarrier",
				"vkCmdPipelineBarrier",
				"vkCmdPipelineBarrier2",
				"vkQueueSubmit"
				});

		if (!looksSync) {
			return VK_FALSE; // Drop non-sync warnings to keep logs clean
		}
	}

	// Optional: filter out noisy best-practices stuff even if it slips in
	if (!isError) {
		if (msg.find("should be sub-allocated") != std::string_view::npos) return VK_FALSE;
		if (msg.find("pipeline cache") != std::string_view::npos) return VK_FALSE;
		if (msg.find("deprecated extension") != std::string_view::npos) return VK_FALSE;
	}

	// De-duplicate messages
	static std::unordered_map<size_t, uint32_t> seen;
	const size_t h = std::hash<std::string_view>{}(msg);
	uint32_t& count = seen[h];
	count++;
	if (count > 1) {
		return VK_FALSE; // print only once
	}

	if (isError)  std::cerr << "[Vulkan][ERROR] " << cmsg << "\n";
	else          std::cerr << "[Vulkan][WARNING] " << cmsg << "\n";

	// Print objects only for errors (and for sync warnings if you want)
	if (cb && cb->objectCount > 0 && cb->pObjects) {
		if (isError /*|| true if you want objects for sync warnings too */) {
			for (uint32_t i = 0; i < cb->objectCount; ++i) {
				const auto& obj = cb->pObjects[i];
				std::cerr
					<< "  - object[" << i << "] type=" << VkObjectType(obj.objectType)
					<< " handle=0x" << std::hex << obj.objectHandle << std::dec
					<< " name=" << (obj.pObjectName ? obj.pObjectName : "(null)")
					<< "\n";
			}
		}
	}

	return VK_FALSE;
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
			bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_4;

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
		// 1: Vulkan 1.4
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