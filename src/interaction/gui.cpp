#pragma once

#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "graphics_context.h"
#include "texture_manager.h"
#include "gpu_sim.h"
#include "cpu_sim.h"

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
		SetRenderingGUI(row, graphicsContext);
		SetSolverTimeingGUI(row, graphicsContext);
		if (graphicsContext.cpu_or_gpu_ == vku::CpuOrGpu::CPU)
			SetSimulationGUI(row, graphicsContext, graphicsContext.cpu_sim_);
		else if (graphicsContext.cpu_or_gpu_ == vku::CpuOrGpu::GPU)
			SetSimulationGUI(row, graphicsContext, graphicsContext.gpu_sim_);
		SetTestSceneGUI(row, graphicsContext.test_scene_);
		//SetObjectGUI(row, graphicsContext.gpu_sim_->ubo_datas_.render);

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

template<typename RowFn, typename Sim>
void GUI::SetSimulationGUI(RowFn&& row, GraphicsContext& graphicsContext, Sim& sim)
{
	if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		ImGui::SeparatorText("PolygonMode");
		const char* items[] = { "Solid", "Wireframe", "Point" };
		int item_current = graphicsContext.polygon_mode_;
		ImGui::ListBox("##", &item_current, items, IM_ARRAYSIZE(items), 3);
		graphicsContext.polygon_mode_ = vku::PolygonMode(item_current);

		ImGui::SeparatorText("Parameter");
		if (ImGui::BeginTable("Parameter", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Nx x Ny", [&] { ImGui::Text("%u x %u", sim->datas_.nx, sim->datas_.ny); });
			row("NumParticles", [&] { ImGui::Text("%u", sim->datas_.num_particles); });
			row("NumEdges", [&] { ImGui::Text("%u", sim->datas_.num_edges); });
			row("NumShears", [&] { ImGui::Text("%u", sim->datas_.num_shears); });
			row("NumBends", [&] { ImGui::Text("%u", sim->datas_.num_bends); });
			row("1 / FrameDt", [&] { ImGui::DragFloat("##FrameDt", &sim->datas_.frame_dt, 1.0f, 60.0f, 240.0f); });
			row("Substeps", [&] { ImGui::DragInt("##Substeps", &sim->datas_.substeps, 1, 1, 40); });
			row("Iterations", [&] { ImGui::DragInt("##Iterations", &sim->datas_.iterations, 1, 1, 40); });
			row("Mass", [&] { ImGui::DragFloat("##Mass", &sim->datas_.mass, 0.001f, 0.0f, 10.0f); });
			row("GlobalDamping", [&] { ImGui::DragFloat("##GlobalDamping", &sim->ubo_.datas.sim_params.global_damping, 0.1f, 1.0f, 2.0f); });
			row("RelaxationFactor", [&] { ImGui::DragFloat("##RelaxationFactor", &sim->ubo_.datas.sim_params.relaxation_factor, 0.1f, 0.0f, 1.0f); });
			row("ClothSize", [&] { ImGui::DragFloat2("##ClothSize", &sim->datas_.cloth_size[0], 0.1f, 0.0f, 10.0f); });
			row("ClothHeight", [&] { ImGui::DragFloat("##ClothHeight", &sim->datas_.cloth_height, 0.1f, 0.0f, 100.0f); });
			row("SelfCollisionStiffness", [&] { ImGui::DragFloat("##SelfCollisionStiffness", &sim->ubo_.datas.sim_params.self_collision_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
			row("Thickness", [&] { ImGui::DragFloat("##Thickness", &sim->ubo_.datas.sim_params.thickness, 0.001f, 0.0f, 1.0f, "%.3f"); });
			row("Friction", [&] { ImGui::DragFloat("##Friction", &sim->ubo_.datas.sim_params.friction, 0.001f, 0.0f, 1.0f, "%.3f"); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Compliance");
		if (ImGui::BeginTable("Compliance", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim->datas_.compliance.stretch, 1e-10f, 0.0f, 1.0f, "%.10f"); });
			row("Shear", [&] { ImGui::DragFloat("##Shear", &sim->datas_.compliance.shear
				, 1e-10f, 0.0f, 1.0f, "%.10f"); });
			row("Bend", [&] { ImGui::DragFloat("##Bend", &sim->datas_.compliance.bend
				, 1e-2f, 0.0f, 1.0f, "%.10f"); });
			row("Area", [&] { ImGui::DragFloat("##Area", &sim->datas_.compliance.area
				, 1e-2f, 0.0f, 1.0f, "%.10f"); });
			row("SelfCollision", [&] { ImGui::DragFloat("##SelfCollision", &sim->datas_.compliance.self_collision
				, 1e-10f, 0.0f, 1.0f, "%.10f"); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Beta");
		if (ImGui::BeginTable("Beta", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim->datas_.beta.stretch, 1.0f, 0.0f, 1000.0f, "%.1f"); });

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

template<typename RowFn>
void GUI::SetRenderingGUI(RowFn&& row, GraphicsContext& graphicsContext)
{
	if (ImGui::CollapsingHeader("Rendering"))//, ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::SeparatorText("SpotLight");
		if (ImGui::BeginTable("SpotLight", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { bool enable = graphicsContext.ubo_datas_.light.light_enable; ImGui::Checkbox("##Enable", &enable); graphicsContext.ubo_datas_.light.light_enable = enable; });
			row("Pos", [&] { ImGui::DragFloat3("##Pos", &graphicsContext.ubo_datas_.light.position[0], 0.1f); });
			row("Dir", [&] { ImGui::DragFloat3("##Dir", &graphicsContext.ubo_datas_.light.direction[0], 0.1f); });
			row("Inner", [&] { ImGui::DragFloat("##Inner", &graphicsContext.ubo_datas_.light.inner, 0.1f); });
			row("Outer", [&] { ImGui::DragFloat("##Outer", &graphicsContext.ubo_datas_.light.outer, 0.1f); });
			row("Intensity", [&] { ImGui::DragFloat("##Intensity", &graphicsContext.ubo_datas_.light.intensity, 0.1f, 0.0f, 100.0f); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("PBR");
		if (ImGui::BeginTable("PBR", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { bool enable = graphicsContext.ubo_datas_.light.pbr_enable; ImGui::Checkbox("##Enable", &enable); graphicsContext.ubo_datas_.light.pbr_enable = enable; });
			row("Exposure", [&] { ImGui::DragFloat("##Exposure", &graphicsContext.ubo_datas_.light.exposure, 0.1f, 0.0f, 2.0f); });
			ImGui::EndTable();
		}
	}
}