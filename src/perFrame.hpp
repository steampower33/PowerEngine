struct PerFrame {
	vk::raii::Fence        inFlight{ nullptr };
	vk::raii::CommandBuffer graphicCmd{ nullptr };
	vk::raii::CommandBuffer computeCmd{ nullptr };

	vk::raii::Buffer        modelUbo{ nullptr };
	vk::raii::DeviceMemory  modelUboMem{ nullptr };
	void* modelUboMapped = nullptr;
	vk::raii::DescriptorSet modelDescriptorSet{ nullptr };

	vk::raii::Buffer		ssbo{ nullptr };
	vk::raii::DeviceMemory  ssboMem{ nullptr };

	vk::raii::Buffer        particleUbo{ nullptr };
	vk::raii::DeviceMemory  particleUboMem{ nullptr };
	void* particleUboMapped = nullptr;
	vk::raii::DescriptorSet particleDescriptorSet{ nullptr };

};