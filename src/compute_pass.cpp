#include "compute_pass.h"
#include "project_types.h"
#include "vulkan_utils.h"

ComputePass::ComputePass(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, Counts& counts)
{
	CreateDescriptorSetLayout(device, counts);
	CreateComputePipeline(device);
    CreateShaderStorageBuffers(physicalDevice, device, queue, commandPool);
    CreateUniformBuffers(physicalDevice, device);
}

void ComputePass::RecordCommands(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame)
{
    cmd.reset();
    cmd.begin({});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0, { descriptor_sets_[currentFrame] }, {});
    cmd.dispatch(PARTICLE_COUNT / 256, 1, 1);
    cmd.end();
}

void ComputePass::UpdateUniformBuffer(uint32_t currentFrame, float dt)
{
    ParticleUBO ubo{};
    ubo.delta_time = 1.0f;

    memcpy(uniform_buffers_mapped_[currentFrame], &ubo, sizeof(ubo));
}

void ComputePass::CreateDescriptorSetLayout(vk::raii::Device& device, Counts& counts) {
	std::array layoutBindings{
		vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
		vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
		vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
	};
	counts.ubo += 1;
	counts.sb += 2;
	counts.layout += MAX_FRAMES_IN_FLIGHT;

	vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
    descriptor_set_layout_ = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void ComputePass::CreateComputePipeline(vk::raii::Device& device)
{
	vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/particle.spv"));

	vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "compMain" };
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*descriptor_set_layout_ };
	pipeline_layout_ = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
	vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layout_ };
	pipeline_ = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}


void ComputePass::CreateShaderStorageBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool) {
    // Initialize particles
    std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution rndDist(0.0f, 1.0f);

    // Initial particle positions on a circle
    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto& particle : particles) {
        float r = 0.25f * sqrtf(rndDist(rndEngine));
        float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
        float x = r * cosf(theta) * 900.0 / 1400.0;
        float y = r * sinf(theta);
        particle.position = glm::vec2(x, y);
        particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
        particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
    }

    vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

    // Create a staging buffer used to upload data to the gpu
    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    vku::CreateBuffer(physicalDevice, device, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, particles.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    shader_storage_buffers_.clear();
    shader_storage_buffers_memory_.clear();

    // Copy initial particle data to all storage buffers
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Buffer shaderStorageBufferTemp({});
        vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
        vku::CreateBuffer(physicalDevice, device, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
        vku::CopyBuffer(device, queue, commandPool, stagingBuffer, shaderStorageBufferTemp, bufferSize);
        shader_storage_buffers_.emplace_back(std::move(shaderStorageBufferTemp));
        shader_storage_buffers_memory_.emplace_back(std::move(shaderStorageBufferTempMemory));
    }
}

void ComputePass::CreateUniformBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device) {
    uniform_buffers_.clear();
    uniform_buffers_memory_.clear();
    uniform_buffers_mapped_.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(ParticleUBO);
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        vku::CreateBuffer(physicalDevice, device, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniform_buffers_.emplace_back(std::move(buffer));
        uniform_buffers_memory_.emplace_back(std::move(bufferMem));
        uniform_buffers_mapped_.emplace_back(uniform_buffers_memory_[i].mapMemory(0, bufferSize));
    }
}

void ComputePass::CreateComputeDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool) {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptor_set_layout_);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    descriptor_sets_.clear();
    descriptor_sets_ = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo(uniform_buffers_[i], 0, sizeof(ParticleUBO));

        vk::DescriptorBufferInfo storageBufferInfoLastFrame(shader_storage_buffers_[(i - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
        vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shader_storage_buffers_[i], 0, sizeof(Particle) * PARTICLE_COUNT);
        std::array descriptorWrites{
            vk::WriteDescriptorSet{.dstSet = *descriptor_sets_[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pImageInfo = nullptr, .pBufferInfo = &bufferInfo, .pTexelBufferView = nullptr },
            vk::WriteDescriptorSet{.dstSet = *descriptor_sets_[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoLastFrame, .pTexelBufferView = nullptr },
            vk::WriteDescriptorSet{.dstSet = *descriptor_sets_[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoCurrentFrame, .pTexelBufferView = nullptr },
        };
        device.updateDescriptorSets(descriptorWrites, {});
    }
}