#pragma once

#include "render_pass.h"

struct Counts;

class ComputePass : public RenderPass {
public:

    ComputePass(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, Counts& counts);
    ~ComputePass() override = default;

    void RecordCommands(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame);
    void UpdateUniformBuffer(uint32_t currentFrame, float dt);

    void CreateComputeDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool);

    std::vector<vk::raii::Buffer> shader_storage_buffers_;
    std::vector<vk::raii::DeviceMemory> shader_storage_buffers_memory_;
private:

    vk::raii::DescriptorSetLayout descriptor_set_layout_ = nullptr;
    vk::raii::PipelineLayout pipeline_layout_ = nullptr;
    vk::raii::Pipeline pipeline_ = nullptr;

    std::vector<vk::raii::Buffer> uniform_buffers_;
    std::vector<vk::raii::DeviceMemory> uniform_buffers_memory_;
    std::vector<void*> uniform_buffers_mapped_;

    std::vector<vk::raii::DescriptorSet> descriptor_sets_;

private:
    void CreateDescriptorSetLayout(vk::raii::Device& device, Counts& counts);
    void CreateComputePipeline(vk::raii::Device& device);
    void CreateShaderStorageBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool);
    void CreateUniformBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device);
};