#pragma once

#include "render_pass.h"

struct Counts;

class GraphicsPass : public RenderPass {
public:

    GraphicsPass(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, Counts& counts, vk::SurfaceFormatKHR& swapchainSurfaceFormat);
    ~GraphicsPass() override = default;

    void RecordCommands(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame, uint32_t imageIndex, vk::Image& image, vk::raii::ImageView& imageView, vk::Extent2D& extent, vk::raii::Buffer& ssbo);

    void CreateGraphicsDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool);
private:
    vk::raii::PipelineLayout pipeline_layout_ = nullptr;
    vk::raii::Pipeline pipeline_ = nullptr;

private:
    void CreateDescriptorSetLayout(vk::raii::Device& device, Counts& counts);
    void CreateGraphicsPipeline(vk::raii::Device& device, vk::SurfaceFormatKHR& swapchainSurfaceFormat);
    void CreateUniformBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device);

    void Transition_image_layout(
        vk::Image& image,
        const vk::raii::CommandBuffer& cmd,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask
    );
};