#include "graphics_pass.h"
#include "project_types.h"
#include "vulkan_utils.h"

GraphicsPass::GraphicsPass(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, Counts& counts, vk::SurfaceFormatKHR& swapchainSurfaceFormat)
{
    CreateDescriptorSetLayout(device, counts);
    CreateGraphicsPipeline(device, swapchainSurfaceFormat);
    CreateUniformBuffers(physicalDevice, device);
}

void GraphicsPass::RecordCommands(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame, uint32_t imageIndex, vk::Image& image, vk::raii::ImageView& imageView, vk::Extent2D& extent, vk::raii::Buffer& ssbo)
{
    cmd.reset();
    cmd.begin({});
    // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    Transition_image_layout(
        image,
        cmd,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},                                                     // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
        vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
    );
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo attachmentInfo = {
        .imageView = imageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = { 0, 0 }, .extent = extent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo
    };
    cmd.beginRendering(renderingInfo);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.bindVertexBuffers(0, { ssbo }, { 0 });
    cmd.draw(PARTICLE_COUNT, 1, 0, 0);

    // Imgui Render
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, *cmd);


    cmd.endRendering();
    // After rendering, transition the swapchain image to PRESENT_SRC
    Transition_image_layout(
        image,
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
        {},                                                      // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
    );
    cmd.end();
}

void GraphicsPass::Transition_image_layout(
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

void GraphicsPass::CreateDescriptorSetLayout(vk::raii::Device& device, Counts& counts) {
    //std::array layoutBindings{
    //    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
    //    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
    //    vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
    //};
    //counts.ubo += 1;
    //counts.sb += 2;
    //counts.layout += MAX_FRAMES_IN_FLIGHT;

    //vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
    //computeDescriptorSetLayout_ = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void GraphicsPass::CreateGraphicsPipeline(vk::raii::Device& device, vk::SurfaceFormatKHR& swapchainSurfaceFormat)
{
    vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/particle.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto bindingDescription = Particle::GetBindingDescription();
    auto attributeDescriptions = Particle::GetAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription, .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data() };
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::ePointList, .primitiveRestartEnable = vk::False };
    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipeline_layout_ = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
      {.stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = pipeline_layout_,
        .renderPass = nullptr },
      {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchainSurfaceFormat.format }
    };

    pipeline_ = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void GraphicsPass::CreateUniformBuffers(vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device) {
    //uniform_buffers_.clear();
    //uniform_buffers_memory_.clear();
    //uniform_buffers_mapped_.clear();

    //for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    //    vk::DeviceSize bufferSize = sizeof(ParticleUBO);
    //    vk::raii::Buffer buffer({});
    //    vk::raii::DeviceMemory bufferMem({});
    //    vku::createBuffer(physicalDevice, device, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
    //    uniform_buffers_.emplace_back(std::move(buffer));
    //    uniform_buffers_memory_.emplace_back(std::move(bufferMem));
    //    uniform_buffers_mapped_.emplace_back(uniform_buffers_memory_[i].mapMemory(0, bufferSize));
    //}
}

void GraphicsPass::CreateGraphicsDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool) {
    //std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout_);
    //vk::DescriptorSetAllocateInfo allocInfo{};
    //allocInfo.descriptorPool = *descriptorPool;
    //allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    //allocInfo.pSetLayouts = layouts.data();
    //computeDescriptorSets.clear();
    //computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

    //for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    //    vk::DescriptorBufferInfo bufferInfo(uniform_buffers_[i], 0, sizeof(ParticleUBO));

    //    vk::DescriptorBufferInfo storageBufferInfoLastFrame(shader_storage_buffers_[(i - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
    //    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shader_storage_buffers_[i], 0, sizeof(Particle) * PARTICLE_COUNT);
    //    std::array descriptorWrites{
    //        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pImageInfo = nullptr, .pBufferInfo = &bufferInfo, .pTexelBufferView = nullptr },
    //        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoLastFrame, .pTexelBufferView = nullptr },
    //        vk::WriteDescriptorSet{.dstSet = *computeDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoCurrentFrame, .pTexelBufferView = nullptr },
    //    };
    //    device.updateDescriptorSets(descriptorWrites, {});
    //}
}


