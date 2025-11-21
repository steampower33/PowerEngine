#pragma once

#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "graphics_context.h"
#include "texture_manager.h"
#include "gpu_sim.h"

#include "gui.h"

GUI::GUI(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain)
{
	// ImGUI DescriptorPool
	std::array<vk::DescriptorPoolSize, 11> imguiPoolSizes{
		vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000},
	};

	vk::DescriptorPoolCreateInfo imguiPoolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 1000,
		.poolSizeCount = static_cast<uint32_t>(imguiPoolSizes.size()),
		.pPoolSizes = imguiPoolSizes.data()
	};
	imgui_pool_ = vk::raii::DescriptorPool(context.device_, imguiPoolInfo);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.DisplaySize.x = swapchain.swapchain_extent_.width;
	io.DisplaySize.y = swapchain.swapchain_extent_.height;

	ImGui::GetStyle().FontScaleMain = 1.5f;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	VkFormat depthFmt = static_cast<VkFormat>(vku::FindDepthFormat(context.physical_device_));
	VkFormat colorFmt = static_cast<VkFormat>(swapchain.swapchain_surface_format_.format);
	static VkFormat colorFormats[] = { colorFmt };
	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);
	ImGui_ImplVulkan_InitInfo init_info = {
		.ApiVersion = vk::ApiVersion14,
		.Instance = *context.instance_,
		.PhysicalDevice = *context.physical_device_,
		.Device = *context.device_,
		.QueueFamily = context.queue_index_,
		.Queue = *context.queue_,
		.DescriptorPool = *imgui_pool_,
		.MinImageCount = swapchain.min_image_count_,
		.ImageCount = swapchain.image_count_,
		.PipelineCache = NULL,
		.PipelineInfoMain = {
			.RenderPass = NULL,
			.Subpass = 0,
			.MSAASamples = static_cast<VkSampleCountFlagBits>(vk::SampleCountFlagBits::e1),
			.PipelineRenderingCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
				.pNext = NULL,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &colorFmt,
				.depthAttachmentFormat = depthFmt,
				.stencilAttachmentFormat = VK_FORMAT_UNDEFINED
			},
		},
		.UseDynamicRendering = true,
		.Allocator = NULL,
	};

	ImGui_ImplVulkan_Init(&init_info);
}

GUI::~GUI()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void GUI::SetStyle()
{
	ImGuiStyle& st = ImGui::GetStyle();
	st.WindowRounding = 12.f;
	st.FrameRounding = 10.f;
	st.GrabRounding = 10.f;
	st.ScrollbarRounding = 10.f;

	ImVec4* col = st.Colors;
	col[ImGuiCol_WindowBg].w = 0.1f;

}

void GUI::Update(Context& context, GraphicsContext& graphicsContext, Swapchain& swapchain)
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	SetStyle();

	auto row = [&](const char* label, auto drawControl)
		{
			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(label);

			ImGui::TableSetColumnIndex(1);
			ImGui::SetNextItemWidth(-FLT_MIN);
			drawControl();
		};

	ImGui::SetNextWindowBgAlpha(1.0f);
	ImGuiWindowFlags wf =
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoCollapse;

	ImGui::SetNextWindowSize(ImVec2(500, 0));
	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Once);
	if (ImGui::Begin("Option", nullptr, wf))
	{
		if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginTable("Rendering", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Exposure", [&] { ImGui::DragFloat("##Exposure", &graphicsContext.ubo_data_.light.exposure, 0.1f, 0.0f, 2.0f); });
				ImGui::EndTable();
			}

		}

		//SetObjectGUI(row, graphicsContext.gpu_sim_->ubo_data_.render);
		//SetLightGUI(row, graphicsContext.ubo_data_);
		SetSolverTimeingGUI(row, graphicsContext);
		SetSimulationGUI(row, graphicsContext);
		SetTestSceneGUI(row, graphicsContext.test_scene_);

		ImGui::End();
	}

	//ImGui::ShowDemoWindow();

	ImGui::Render();
}

void GUI::DisplayKernelTiming(const std::string name, std::unordered_map<std::string, double>& labelToTime, std::unordered_map<std::string, double>& labelToAvgTime, bool autoColor)
{
	double total = labelToAvgTime["Total"];
	float percentage = (total > 0) ? float(labelToAvgTime[name] / total * 100) : 0.0f;

	bool shouldPop = false;

	if (autoColor)
	{
		if (percentage > 10 || percentage == 0.0f)
		{
			ImVec4 textColor = color_low;
			if (percentage > 30)
				textColor = color_high;
			else if (percentage > 10)
				textColor = color_mid;
			else if (percentage == 0.0f)
				textColor = color_disabled;
			ImGui::PushStyleColor(ImGuiCol_Text, textColor);
			shouldPop = true;
		}
	}

	auto RightText = [&](const char* text)
		{
			float w = ImGui::GetColumnWidth();
			float tw = ImGui::CalcTextSize(text).x;
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (w - tw));
			ImGui::TextUnformatted(text);
		};

	ImGui::TableNextColumn();
	ImGui::Text(name.c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", labelToTime[name]).c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", labelToAvgTime[name] / count_).c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", percentage).c_str());

	if (shouldPop)
	{
		ImGui::PopStyleColor();
	}
}

template<typename RowFn, typename UBOData>
void GUI::SetLightGUI(RowFn&& row, UBOData& data) {
	if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
	{

		if (ImGui::BeginTable("LightTable", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Pos", [&] { ImGui::DragFloat3("##Pos", &data.light.spotPos_range[0], 0.1f); });
			row("Range", [&] { ImGui::DragFloat("##Range", &data.light.spotPos_range[3], 0.1f); });
			row("Dir", [&] { ImGui::DragFloat3("##Dir", &data.light.spotDir_inner[0], 0.1f); });
			row("Color", [&] { ImGui::DragFloat3("##Color", &data.light.spotColor_outer[0], 0.1f); });
			row("Inner", [&] { ImGui::DragFloat("##Inner", &data.light.spotDir_inner[3], 0.1f); });
			row("Outer", [&] { ImGui::DragFloat("##Outer", &data.light.spotColor_outer[3], 0.1f); });


			ImGui::EndTable();
		}
	}
}

template<typename RowFn, typename UBOData>
void GUI::SetObjectGUI(RowFn&& row, UBOData& data) {

	if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::BeginTable("ObjectTable", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Meltallic", [&] { ImGui::DragFloat("##ClothMeltallic", &data.metallic_factor, 0.1f, 0.0f, 1.0f); });
			row("Roughness", [&] { ImGui::DragFloat("##ClothRoughness", &data.roughness_factor, 0.1f, 0.0f, 1.0f); });
			row("AO", [&] { ImGui::DragFloat("##ClothAO", &data.ao_factor, 0.1f, 0.0f, 1.0f); });
			row("Height", [&] { ImGui::DragFloat("##ClothHeight", &data.height_factor, 0.001f, 0.0f, 1.0f); });

			row("AlbedoEnable", [&] {
				bool enable = (data.albedo_enable == 1) ? true : false;
				ImGui::Checkbox("##ClothAlbedoEnable", &enable);
				data.albedo_enable = (enable) ? true : false;
				});
			row("MetalnessEnable", [&] {
				bool enable = (data.metallic_enable == 1) ? true : false;
				ImGui::Checkbox("##ClothMetalnessEnable", &enable);
				data.metallic_enable = (enable) ? true : false;
				});
			row("NormalEnable", [&] {
				bool enable = (data.normal_enable == 1) ? true : false;
				ImGui::Checkbox("##ClothNormalEnable", &enable);
				data.normal_enable = (enable) ? true : false;
				});
			row("RoughtnessEnable", [&] {
				bool enable = (data.roughtnessEnable == 1) ? true : false;
				ImGui::Checkbox("##ClothRoughtnessEnable", &enable);
				data.roughtnessEnable = (enable) ? true : false;
				});
			row("AOEnable", [&] {
				bool enable = (data.ao_enable == 1) ? true : false;
				ImGui::Checkbox("##ClothAOEnable", &enable);
				data.ao_enable = (enable) ? true : false;
				});
			row("HeightEnable", [&] {
				bool enable = (data.height_enable == 1) ? true : false;
				ImGui::Checkbox("##ClothHeightEnable", &enable);
				data.height_enable = (enable) ? true : false;
				});

			ImGui::EndTable();
		}
	}
}

template<typename RowFn>
void GUI::SetSolverTimeingGUI(RowFn&& row, GraphicsContext& graphicsContext)
{
	auto& labels = graphicsContext.labels_;
	auto& labelToTime = graphicsContext.label_time_;
	auto& labelToAvgTime = graphicsContext.label_avg_time_;

	is_print_timestamps = ImGui::CollapsingHeader("Solver timing"); //, ImGuiTreeNodeFlags_DefaultOpen
	if (is_print_timestamps)
	{
		if (ImGui::BeginTable("timing", 4))//,  ImGuiTableFlags_BordersOuter))
		{
			ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch, 0.5f);
			ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
			ImGui::TableSetupColumn("Avg (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
			ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthFixed, 100.0f);
			ImGui::TableHeadersRow();

			for (uint32_t i = 0; i < labels.size() - 1; i++)
			{
				DisplayKernelTiming(labels[i], labelToTime, labelToAvgTime);
			}

			DisplayKernelTiming(labels[labels.size() - 1], labelToTime, labelToAvgTime, false);

			ImGui::EndTable();
		}
		count_ = graphicsContext.time_count_;
	}
}

template<typename RowFn>
void GUI::SetSimulationGUI(RowFn&& row, GraphicsContext& graphicsContext)
{
	if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		const char* items[] = { "Solid", "Wireframe", "Point" };
		int item_current = graphicsContext.polygon_mode_;
		ImGui::ListBox("PolygonMode", &item_current, items, IM_ARRAYSIZE(items), 3);
		graphicsContext.polygon_mode_ = vku::PolygonMode(item_current);

		if (ImGui::BeginTable("SimulationTable", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Nx x Ny", [&] { ImGui::Text("%u x %u", graphicsContext.gpu_sim_->nx_, graphicsContext.gpu_sim_->ny_); });
			row("NumParticles", [&] { ImGui::Text("%u", graphicsContext.gpu_sim_->particles_size_); });
			row("NumEdges", [&] { ImGui::Text("%u", graphicsContext.gpu_sim_->edge_size_); });
			row("NumShears", [&] { ImGui::Text("%u", graphicsContext.gpu_sim_->shear_size_); });
			row("NumBends", [&] { ImGui::Text("%u", graphicsContext.gpu_sim_->bend_size_); });
			row("FrameDt", [&] { ImGui::DragFloat("##FrameDt", &graphicsContext.gpu_sim_->frame_dt_, 1.0f, 1.0f, 240.0f); });
			row("Substeps", [&] { ImGui::DragInt("##Substeps", &graphicsContext.gpu_sim_->substeps_, 1, 1, 40); });
			row("Iterations", [&] { ImGui::DragInt("##Iterations", &graphicsContext.gpu_sim_->iterations_, 1, 1, 40); });
			row("Mass", [&] { ImGui::DragFloat("##Mass", &graphicsContext.gpu_sim_->mass_, 0.001f, 0.0f, 10.0f); });
			row("Damping", [&] { ImGui::DragFloat("##Damping", &graphicsContext.gpu_sim_->ubo_data_.sim_params.damping, 0.1f, 0.0f, 3.0f, "%.2f"); });
			row("RelaxationFactor", [&] { ImGui::DragFloat("##RelaxationFactor", &graphicsContext.gpu_sim_->ubo_data_.sim_params.relaxation_factor, 0.001f, 0.0f, 1.0f, "%.4f"); });
			row("StretchCompliance", [&] { ImGui::DragFloat("##StretchCompliance", &graphicsContext.gpu_sim_->compliance_.stretch, 1e-8f, 0.0f, 1.0f, "%.8f"); });
			row("DiagonalCompliance", [&] { ImGui::DragFloat("##DiagonalCompliance", &graphicsContext.gpu_sim_->compliance_.diagonal, 1e-8f, 0.0f, 1.0f, "%.8f"); });
			row("ShearCompliance", [&] { ImGui::DragFloat("##ShearCompliance", &graphicsContext.gpu_sim_->compliance_.shear
				, 1e-6f, 0.0f, 1.0f, "%.8f"); });
			row("BendCompliance", [&] { ImGui::DragFloat("##BendCompliance", &graphicsContext.gpu_sim_->compliance_.bend
				, 1e-6f, 0.0f, 1.0f, "%.8f"); });


			row("ClothSize", [&] { ImGui::DragFloat2("##ClothSize", &graphicsContext.gpu_sim_->cloth_size_[0], 0.1f, 0.0f, 10.0f); });
			row("ClothHeight", [&] { ImGui::DragFloat("##ClothHeight", &graphicsContext.gpu_sim_->cloth_height_, 0.1f, 0.0f, 100.0f); });

			row("CollisionMargin", [&] { ImGui::DragFloat("##CollisionMargin", &graphicsContext.gpu_sim_->ubo_data_.sim_params.collision_margin, 0.01f, 0.0f, 1.0f, "%.3f"); });
			row("Thickness", [&] { ImGui::DragFloat("##Thickness", &graphicsContext.gpu_sim_->ubo_data_.sim_params.thickness, 0.001f, 0.0f, 1.0f, "%.3f"); });
			row("Friction", [&] { ImGui::DragFloat("##Friction", &graphicsContext.gpu_sim_->ubo_data_.sim_params.friction, 0.001f, 0.0f, 1.0f, "%.3f"); });

			bool windEnabled = (graphicsContext.gpu_sim_->ubo_data_.sim_params.windTest != 0);
			row("Wind", [&] {
				if (ImGui::Checkbox("##Wind", &windEnabled)) {
					graphicsContext.gpu_sim_->ubo_data_.sim_params.windTest = windEnabled ? 1 : 0;
				} });

				row("Wind Dir", [&] {
					ImGui::DragFloat3("##WindDir", &graphicsContext.gpu_sim_->ubo_data_.sim_params.wind_dir[0], 0.1f, -1.0f, 1.0f); });

				row("Wind Strength", [&] {
					ImGui::DragFloat("##WindStrength", &graphicsContext.gpu_sim_->ubo_data_.sim_params.wind_strength, 0.1f, 0.0f, 5.0f); });

				ImGui::EndTable();
		}
	}
}


template<typename RowFn, typename Scene>
void GUI::SetTestSceneGUI(RowFn&& row, Scene& scene)
{
	if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::BeginTable("SceneTable", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("SphereCollision", [&] {
				ImGui::Checkbox("##SphereCollision", &scene.sphereCollision); });
			row("PinnedCorner", [&] {
				ImGui::Checkbox("##PinnedCorner", &scene.pinnedCorner); });
			row("TopPinnedCorner", [&] {
				ImGui::Checkbox("##TopPinnedCorner", &scene.topPinnedCorner); });

			ImGui::EndTable();
		}
	}
}