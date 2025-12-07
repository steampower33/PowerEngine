#pragma once

#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "pass_manager.h"
#include "simulation_pass_gpu.h"
#include "simulation_pass_cpu.h"
#include "model.h"
#include "model_manager.h"
#include "texture_manager.h"
#include "graphics_pass.h"
#include "particle_manager.h"
#include "camera.h"

#include "gui.h"

GUI::GUI(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager, PassManager& passManager)
	: context_(context), swapchain_(swapchain), texture_manager_(textureManager), model_manager_(modelManager), pass_manager_(passManager)
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
	imgui_pool_ = vk::raii::DescriptorPool(context_.device_, imguiPoolInfo);

	// Setup Dear ImGui context_
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.DisplaySize.x = swapchain.swapchain_extent_.width;
	io.DisplaySize.y = swapchain.swapchain_extent_.height;

	// Setup Platform/Renderer backends
	VkFormat depthFmt = static_cast<VkFormat>(vku::FindDepthFormat(context_.physical_device_));
	VkFormat colorFmt = static_cast<VkFormat>(swapchain.swapchain_surface_format_.format);
	static VkFormat colorFormats[] = { colorFmt };
	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);
	ImGui_ImplVulkan_InitInfo init_info = {
		.ApiVersion = vk::ApiVersion14,
		.Instance = *context_.instance_,
		.PhysicalDevice = *context_.physical_device_,
		.Device = *context_.device_,
		.QueueFamily = context_.queue_index_,
		.Queue = *context_.queue_,
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
	//ImGui::StyleColorsClassic();
	//ImGui::StyleColorsLight();
	//ImGui::StyleColorsDark();

	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowRounding = 8.0f;
	style.FrameRounding = 4.0f;
	style.GrabRounding = 4.0f;
	style.ScrollbarRounding = 4.0f;
	style.FontScaleMain = 2.0f;

	ImVec4* colors = style.Colors;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.04f, 0.04f, 0.04f, 0.99f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.38f, 0.51f, 0.51f, 0.80f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.03f, 0.03f, 0.04f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.01f, 0.01f, 0.01f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.07f, 0.07f, 0.07f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.18f, 0.17f, 0.17f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.30f, 0.60f, 0.10f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.60f, 0.10f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.43f, 0.90f, 0.11f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.21f, 0.22f, 0.23f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.51f, 0.51f, 0.80f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.54f, 0.55f, 0.55f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.38f, 0.51f, 0.51f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.03f, 0.03f, 0.03f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.16f, 0.16f, 0.16f, 0.50f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.20f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.23f, 0.23f, 0.24f, 0.80f);
	colors[ImGuiCol_Tab] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_TabSelected] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_TabSelectedOverline] = ImVec4(0.13f, 0.78f, 0.07f, 1.00f);
	colors[ImGuiCol_TabDimmed] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_TabDimmedSelected] = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
	colors[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(0.10f, 0.60f, 0.12f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.14f, 0.87f, 0.05f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.30f, 0.60f, 0.10f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.23f, 0.78f, 0.02f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.27f, 0.27f, 0.27f, 1.00f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.46f, 0.47f, 0.46f, 0.06f);
	colors[ImGuiCol_TextLink] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavCursor] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.78f, 0.69f, 0.69f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

	style.WindowRounding = 4.0f;
	style.FrameRounding = 4.0f;
	style.GrabRounding = 3.0f;
	style.PopupRounding = 4.0f;
	style.TabRounding = 4.0f;
	style.WindowMenuButtonPosition = ImGuiDir_Right;
	style.ScrollbarSize = 10.0f;
	style.GrabMinSize = 10.0f;
	style.SeparatorTextBorderSize = 2.0f;
}

void GUI::Update(float& targetSimFPS, double& simDt, Camera& camera)
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

	ImGuiWindowFlags wf =
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove;

	float spacing = 10.0f;

	// Left Side
	ImVec2 scenePos{ spacing, spacing };
	ImVec2 sceneSize{ 500.0f, 200.0f };

	ImGui::SetNextWindowPos(scenePos);
	ImGui::SetNextWindowSize(sceneSize);
	if (ImGui::Begin("Scene", nullptr, wf))
	{
		SetTestSceneGUI(row, pass_manager_.test_scene_);
	}
	ImGui::End();

	ImVec2 optionPos{ scenePos.x, scenePos.y + sceneSize.y + spacing };
	ImVec2 optionSize{ sceneSize.x, swapchain_.swapchain_extent_.height - optionPos.y - spacing };
	ImGui::SetNextWindowPos(optionPos);
	ImGui::SetNextWindowSize(optionSize);
	if (ImGui::Begin("Option", nullptr, wf))
	{
		SetCameraGUI(row, camera);
		SetRenderingGUI(row);

		SetObjectsGUI(row, model_manager_.models_, pass_manager_.graphics_pass_->ubo_datas_.cloth);

		if (pass_manager_.cpu_or_gpu_ == vku::CpuOrGpu::GPU)
			SetSimulationGUI(row, *pass_manager_.sim_pass_gpu_, targetSimFPS, simDt);

	}
	ImGui::End();

	// Right Side
	ImGui::SetNextWindowSize(ImVec2(600.0f, 0));
	ImGui::SetNextWindowPos(ImVec2(swapchain_.swapchain_extent_.width - 610.0f, spacing));
	if (ImGui::Begin("Cloth Performance Monitor", nullptr, wf))
	{
		SetStatGUI(row);

		SetTimeingGUI(row, *pass_manager_.sim_pass_gpu_);
	}
	ImGui::End();

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

template<typename RowFn, typename Objects, typename ClothUBO>
void GUI::SetObjectsGUI(RowFn&& row, Objects& objects, ClothUBO& clothUBO) {
	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen))
	{
		for (auto& object : objects)
		{
			if (ImGui::TreeNode(object->name_.c_str()))
			{
				ImGui::SeparatorText("Factor");
				if (ImGui::BeginTable("Factor", 2,
					ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
				{
					row("Albedo", [&] { ImGui::DragFloat3("##Albedo", &object->albedo_[0], 0.1f, 0.0f, 1.0f); });
					row("Meltallic", [&] { ImGui::DragFloat("##Meltallic", &object->factors_.metallic, 0.1f, 0.0f, 1.0f); });
					row("Roughness", [&] { ImGui::DragFloat("##Roughness", &object->factors_.roughness, 0.1f, 0.0f, 1.0f); });
					row("AO", [&] { ImGui::DragFloat("##AO", &object->factors_.ao, 0.1f, 0.0f, 1.0f); });
					row("Height", [&] { ImGui::DragFloat("##Height", &object->factors_.height, 0.001f, 0.0f, 1.0f); });
					row("SheenWeight", [&] { ImGui::DragFloat("##SheenWeight", &object->factors_.sheen_weight, 0.001f, 0.0f, 1.0f); });
					row("SheenRoughness", [&] { ImGui::DragFloat("##SheenRoughness", &object->factors_.sheen_roughness, 0.001f, 0.0f, 1.0f); });
					ImGui::EndTable();
				}

				ImGui::SeparatorText("Enable");
				if (ImGui::BeginTable("Enable", 2,
					ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
				{
					row("Albedo", [&] { ImGui::Checkbox("##Albedo", &object->texture_enable_.albedo); });
					row("Meltallic", [&] { ImGui::Checkbox("##Meltallic", &object->texture_enable_.metallic); });
					row("Normal", [&] { ImGui::Checkbox("##Normal", &object->texture_enable_.normal); });
					row("Roughtness", [&] { ImGui::Checkbox("##Roughtness", &object->texture_enable_.roughness); });
					row("AO", [&] { ImGui::Checkbox("##AO", &object->texture_enable_.ao); });
					row("Height", [&] { ImGui::Checkbox("##Height", &object->texture_enable_.height); });
					row("CheckerBoard", [&] { ImGui::Checkbox("##CheckerBoard", &object->checker_board_enable_); });
					row("Movable", [&] { ImGui::Checkbox("##Movable", &object->movable_); });
					ImGui::EndTable();
				}
				ImGui::TreePop();
			}
		}

		if (ImGui::TreeNode("Cloth"))
		{
			ImGui::SeparatorText("Factor");
			if (ImGui::BeginTable("Factor", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Albedo", [&] { ImGui::DragFloat3("##Albedo", &clothUBO.albedo[0], 0.1f, 0.0f, 1.0f); });
				row("Meltallic", [&] { ImGui::DragFloat("##Meltallic", &clothUBO.metallic_factor, 0.1f, 0.0f, 1.0f); });
				row("Roughness", [&] { ImGui::DragFloat("##Roughness", &clothUBO.roughness_factor, 0.1f, 0.0f, 1.0f); });
				row("AO", [&] { ImGui::DragFloat("##AO", &clothUBO.ao_factor, 0.1f, 0.0f, 1.0f); });
				row("Height", [&] { ImGui::DragFloat("##Height", &clothUBO.height_factor, 0.001f, 0.0f, 1.0f); });
				row("SheenWeight", [&] { ImGui::DragFloat("##SheenWeight", &clothUBO.sheen_weight_factor, 0.001f, 0.0f, 1.0f); });
				row("SheenRoughness", [&] { ImGui::DragFloat("##SheenRoughness", &clothUBO.sheen_roughness_factor, 0.001f, 0.0f, 1.0f); });
				ImGui::EndTable();
			}

			ImGui::SeparatorText("Enable");
			if (ImGui::BeginTable("Enable", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Albedo", [&] { bool check = clothUBO.albedo_enable; ImGui::Checkbox("##Albedo", &check); clothUBO.albedo_enable = check; });
				row("Meltallic", [&] { bool check = clothUBO.metallic_enable;  ImGui::Checkbox("##Meltallic", &check); clothUBO.metallic_enable = check;  });
				row("Normal", [&] { bool check = clothUBO.normal_enable; ImGui::Checkbox("##Normal", &check); clothUBO.normal_enable = check; });
				row("Roughtness", [&] { bool check = clothUBO.roughness_enable; ImGui::Checkbox("##Roughtness", &check); clothUBO.roughness_enable = check; });
				row("AO", [&] { bool check = clothUBO.ao_enable; ImGui::Checkbox("##AO", &check); clothUBO.ao_enable = check; });
				row("Height", [&] { bool check = clothUBO.height_enable; ImGui::Checkbox("##Height", &check); clothUBO.height_enable = check;  });
				ImGui::EndTable();
			}
			ImGui::TreePop();
		}
	}
}

template<typename RowFn>
void GUI::SetTimeingGUI(RowFn&& row, SimulationPassGPU& sim)
{
	auto& labels = sim.labels_;
	auto& labelToTime = sim.label_time_;
	auto& labelToAvgTime = sim.label_avg_time_;

	open_timestamps_ = ImGui::CollapsingHeader("Timing"); //, ImGuiTreeNodeFlags_DefaultOpen
	if (open_timestamps_)
	{
		if (ImGui::BeginTable("Timing", 4))//,  ImGuiTableFlags_BordersOuter))
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
		ImGui::SeparatorText("Overall");
		ImGui::Text("Compute : %.3f", sim.pass_total_time_);
		ImGui::Text("Graphics : %.3f", pass_manager_.graphics_pass_->pass_total_time_);

		count_ = sim.time_count_;
	}
}

template<typename RowFn, typename Sim>
void GUI::SetSimulationGUI(RowFn&& row, Sim& sim, float& targetSimFPS, double& simDt)
{
	if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		auto& pm = *pass_manager_.particle_manager_;

		ImGui::SeparatorText("Parameter");
		if (ImGui::BeginTable("Parameter", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("TargetSimFPS", [&] { ImGui::DragFloat("##TargetSimFPS", &targetSimFPS, 1.0f, 30.0f, 1000.0f); simDt = 1.0 / static_cast<double>(targetSimFPS); });
			row("1 / FrameDt", [&] { ImGui::DragFloat("##FrameDt", &sim.datas_.frame_dt, 1.0f, 60.0f, 240.0f); });
			row("Substeps", [&] { ImGui::DragInt("##Substeps", &sim.datas_.substeps, 1, 1, 40); });
			row("Iterations", [&] { ImGui::DragInt("##Iterations", &sim.datas_.iterations, 1, 1, 40); });
			row("BroadphaseInterval", [&] { ImGui::DragInt("##BroadphaseInterval", &sim.broadphase_interval_, 1.0f, 0.0f, sim.datas_.substeps); });
			row("NarrowphaseInterval", [&] { ImGui::DragInt("##NarrowphaseInterval", &sim.narrowphase_interval_, 1.0f, 0.0f, sim.datas_.iterations); });
			row("GlobalDamping", [&] { ImGui::DragFloat("##GlobalDamping", &sim.ubo_.datas.sim_params.global_damping, 0.1f, 1.0f, 2.0f); });
			row("RelaxationFactor", [&] { ImGui::DragFloat("##RelaxationFactor", &sim.ubo_.datas.sim_params.relaxation_factor, 0.1f, 0.0f, 1.0f); });
			row("Thickness", [&] { ImGui::DragFloat("##Thickness", &sim.ubo_.datas.sim_params.thickness, 0.001f, 0.0f, 1.0f, "%.3f"); });
			row("Friction", [&] { ImGui::DragFloat("##Friction", &sim.ubo_.datas.sim_params.friction, 0.001f, 0.0f, 1.0f, "%.3f"); });
			row("NeighborFriction", [&] { ImGui::DragFloat("##NeighborFriction", &sim.ubo_.datas.sim_params.neighbor_friction, 0.1f, 0.0f, 10.0f, "%.1f"); });
			row("MaxNeighbors", [&] { int maxNeighbors = sim.ubo_.datas.sim_params.max_neighbors;  ImGui::DragInt("##MaxNeighbors", &maxNeighbors, 1, 0, 20); sim.ubo_.datas.sim_params.max_neighbors = maxNeighbors; });
			row("MaxSpeed", [&] { ImGui::DragFloat("##MaxSpeed", &sim.ubo_.datas.sim_params.max_speed, 0.1f, sim.ubo_.datas.sim_params.max_speed, sim.ubo_.datas.sim_params.max_speed, "%.1f"); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("SolverConfig");
		if (ImGui::BeginTable("SolverConfig", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Stretch", [&] { ImGui::Checkbox("##Stretch", &sim.solver_config_.stretch); });
			row("Shear", [&] { ImGui::Checkbox("##Shear", &sim.solver_config_.shear); });
			row("Bend", [&] { ImGui::Checkbox("##Bend", &sim.solver_config_.bend); });
			row("Area", [&] { ImGui::Checkbox("##Area", &sim.solver_config_.area); });
			row("SelfCollision", [&] { ImGui::Checkbox("##SelfCollision", &sim.solver_config_.self_collision); });
			ImGui::EndTable();
		}


		ImGui::SeparatorText("Stiffness");
		if (ImGui::BeginTable("Stiffness", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("StretchStiffness", [&] { ImGui::DragFloat("##StretchStiffness", &sim.ubo_.datas.sim_params.stretch_stiffness, 1e-2f, 0.0f, 100.0f, "%.3f"); });
			row("SoftbodyStretchStiffness", [&] { ImGui::DragFloat("##SoftbodyStretchStiffness", &sim.ubo_.datas.sim_params.softbody_stretch_stiffness, 1e-2f, 0.0f, 100.0f, "%.1f"); });
			row("SoftbodyVolumeStiffness", [&] { ImGui::DragFloat("##SoftbodyVolumeStiffness", &sim.ubo_.datas.sim_params.volume_stiffness, 1e-2f, 0.0f, 100.0f, "%.1f"); });
			row("ShearStiffness", [&] { ImGui::DragFloat("##ShearStiffness", &sim.ubo_.datas.sim_params.shear_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
			row("BendStiffness", [&] { ImGui::DragFloat("##BendStiffness", &sim.ubo_.datas.sim_params.bend_stiffness, 1e-3f, 0.0f, 2.0f, "%.3f"); });
			row("AreaStiffness", [&] { ImGui::DragFloat("##AreaStiffness", &sim.ubo_.datas.sim_params.area_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
			row("SelfCollisionStiffness", [&] { ImGui::DragFloat("##SelfCollisionStiffness", &sim.ubo_.datas.sim_params.self_collision_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Compliance");
		if (ImGui::BeginTable("Compliance", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim.datas_.compliance.stretch,
				1e-9f, 0.0f, 1.0f, "%.9f"); });
			row("SoftbodyStretch", [&] { ImGui::DragFloat("##SoftbodyStretch", &sim.datas_.compliance.softbody_stretch,
				1e-9f, 0.0f, 1.0f, "%.9f"); });
			row("SoftbodyVolume", [&] { ImGui::DragFloat("##SoftbodyVolume", &sim.datas_.compliance.softbody_volume,
				1e-9f, 0.0f, 1.0f, "%.9f"); });
			row("Shear", [&] { ImGui::DragFloat("##Shear", &sim.datas_.compliance.shear
				, 1e-9f, 0.0f, 1.0f, "%.9f"); });
			row("Bend", [&] { ImGui::DragFloat("##Bend", &sim.datas_.compliance.bend
				, 1e-2f, 0.0f, 1.0f, "%.9f"); });
			row("Area", [&] { ImGui::DragFloat("##Area", &sim.datas_.compliance.area
				, 1e-2f, 0.0f, 1.0f, "%.9f"); });
			row("SelfCollision", [&] { ImGui::DragFloat("##SelfCollision", &sim.datas_.compliance.self_collision
				, 1e-9f, 0.0f, 1.0f, "%.9f"); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Beta");
		if (ImGui::BeginTable("Beta", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim.datas_.beta.stretch, 1.0f, 0.0f, 1000.0f, "%.1f"); });

			ImGui::EndTable();
		}
	}
}

template<typename RowFn, typename Scene>
void GUI::SetTestSceneGUI(RowFn&& row, Scene& scene)
{
	ImVec2 buttonSize = ImVec2(ImGui::GetContentRegionAvail().x, 0);
	if (ImGui::Button("SphereCollision", buttonSize))
	{
		scene.sphereCollision = true;
	}

	if (ImGui::Button("PinnedCorner", buttonSize))
	{
		scene.pinnedCorner = true;
	}

	if (ImGui::Button("TopPinnedCorner", buttonSize))
	{
		scene.topPinnedCorner = true;
	}

	if (ImGui::Button("SelfCollision", buttonSize))
	{
		scene.selfCollision = true;
	}


}

template<typename RowFn>
void GUI::SetRenderingGUI(RowFn&& row)
{

	if (ImGui::CollapsingHeader("Rendering"))//, ImGuiTreeNodeFlags_DefaultOpen))
	{
		auto& gp = *pass_manager_.graphics_pass_;
		auto& tm = texture_manager_;

		ImGui::SeparatorText("PolygonMode");
		const char* items[] = { "Solid", "Wireframe", "Point" };
		int item_current = gp.polygon_mode_;
		ImGui::ListBox("##", &item_current, items, IM_ARRAYSIZE(items), 3);
		gp.polygon_mode_ = vku::PolygonMode(item_current);

		ImGui::SeparatorText("SpotLight");
		if (ImGui::BeginTable("SpotLight", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { bool enable = gp.ubo_datas_.light.light_enable; ImGui::Checkbox("##Enable", &enable); gp.ubo_datas_.light.light_enable = enable; });
			row("Pos", [&] { ImGui::DragFloat3("##Pos", &gp.ubo_datas_.light.position[0], 0.1f); });
			row("Dir", [&] { ImGui::DragFloat3("##Dir", &gp.ubo_datas_.light.direction[0], 0.1f); });
			row("Inner", [&] { ImGui::DragFloat("##Inner", &gp.ubo_datas_.light.inner, 0.1f); });
			row("Outer", [&] { ImGui::DragFloat("##Outer", &gp.ubo_datas_.light.outer, 0.1f); });
			row("Intensity", [&] { ImGui::DragFloat("##Intensity", &gp.ubo_datas_.light.intensity, 0.1f, 0.0f, 100.0f); });

			ImGui::EndTable();
		}

		ImGui::SeparatorText("PBR");
		if (ImGui::BeginTable("PBR", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { bool enable = gp.ubo_datas_.light.pbr_enable; ImGui::Checkbox("##Enable", &enable); gp.ubo_datas_.light.pbr_enable = enable; });
			row("Exposure", [&] { ImGui::DragFloat("##Exposure", &gp.ubo_datas_.light.exposure, 0.1f, 0.0f, 2.0f); });
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Skybox");
		ImVec2 buttonSize = ImVec2(ImGui::GetContentRegionAvail().x, 0);
		if (ImGui::Button("Morning", buttonSize))
		{
			tm.skybox_enable_.morning = true;
		}
		if (ImGui::Button("Evening", buttonSize))
		{
			tm.skybox_enable_.evening = true;
		}
		if (ImGui::Button("Night", buttonSize))
		{
			tm.skybox_enable_.night = true;
		}
	}
}

template<typename RowFn>
void GUI::SetStatGUI(RowFn&& row)
{
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

	auto& pm = *pass_manager_.particle_manager_;
	auto& d = pass_manager_.sim_pass_gpu_->datas_;

	if (ImGui::BeginTable("Stat", 2, ImGuiTableFlags_BordersInnerV))
	{
		row("NumParticles", [&] { ImGui::Text("%u", pm.total_particles_); });
		row("NumEdges", [&] { ImGui::Text("%u", d.num_edges); });
		row("NumShears", [&] { ImGui::Text("%u", d.num_shears); });
		row("NumBends", [&] { ImGui::Text("%u", d.num_bends); });
		row("NumAreas", [&] { ImGui::Text("%u", d.num_areas); });

		ImGui::EndTable();
	}

}

template<typename RowFn>
void GUI::SetCameraGUI(RowFn&& row, Camera& camera)
{

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::BeginTable("Camera", 2, ImGuiTableFlags_BordersInnerV))
		{
			row("MouseMove(R)", [&] { ImGui::Checkbox("##MouseMove", &camera.mouse_move_enabled); });
			row("CameraMoveSpeed", [&] { ImGui::DragFloat("##CameraMoveSpeed", &camera.camera_move_speed, 0.1f, 0.0f, 10.0f); });
			row("MouseMoveSpeed", [&] { ImGui::DragFloat("##MouseMoveSpeed", &camera.mouse_move_speed, 0.1f, 0.0f, 10.0f); });

			row("Focus(F)", [&] { ImGui::Checkbox("##Focus", &camera.focus_enabled); });
			row("OrbitYawSpeed", [&] { ImGui::DragFloat("##OrbitYawSpeed", &camera.orbit_yaw_speed, 0.1f, 0.0f, 360.0f); });
			row("OrbitPitchSpeed", [&] { ImGui::DragFloat("##OrbitPitchSpeed", &camera.orbit_pitch_speed, 0.1f, 0.0f, 360.0f); });
			row("ZoomSpeed", [&] { ImGui::DragFloat("##ZoomSpeed", &camera.zoom_speed, 0.1f, 0.0f, 10.0f); });

			ImGui::EndTable();
		}

	}
}