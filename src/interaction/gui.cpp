#pragma once

#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "pass_manager.h"
#include "simulation_pass_gpu.h"
#include "simulation_pass_cpu.h"
#include "model.h"
#include "model_manager.h"
#include "texture.h"
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

	for (auto& texPtr : texture_manager_.tex2d_)
	{
		if (!texPtr) continue;
		auto& tex = *texPtr;

		VkImageView view = static_cast<VkImageView>(*tex.texture_image_view_);
		VkSampler  sampler = static_cast<VkSampler>(*tex.texture_sampler_);

		if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE)
			continue;

		imgui_id_.push_back(std::pair{ texPtr->filename_, ImGui_ImplVulkan_AddTexture(
			sampler,
			view,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		) });
	}


	VkImageView view = static_cast<VkImageView>(*pass_manager_.graphics_pass_->shadow_image_view_);
	VkSampler  sampler = static_cast<VkSampler>(*pass_manager_.graphics_pass_->shadow_sampler);

	imgui_id_.push_back(std::pair{ "Shadow", ImGui_ImplVulkan_AddTexture(
		sampler,
		view,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	) });
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
	style.FontScaleMain = 1.6f * swapchain_.swapchain_extent_.width / 2560.0f;

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

void GUI::Update(float& targetSimFPS, double& simDt, Camera& camera, bool& paused, bool& pauseEachframe)
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

	float spacing = swapchain_.swapchain_extent_.width * 0.004f;

	// Left Side
	ImVec2 sceneGuiPos{ spacing, spacing };
	ImVec2 sceneGuiSize{ swapchain_.swapchain_extent_.width * 0.2f, swapchain_.swapchain_extent_.height * 0.15f };

	ImGui::SetNextWindowPos(sceneGuiPos);
	ImGui::SetNextWindowSize(sceneGuiSize);
	if (ImGui::Begin("Scene", nullptr, wf))
	{
		SetTestSceneGUI(row);
	}
	ImGui::End();

	ImVec2 optionPos{ sceneGuiPos.x, sceneGuiPos.y + sceneGuiSize.y + spacing };
	ImVec2 optionSize{ sceneGuiSize.x, swapchain_.swapchain_extent_.height - optionPos.y - spacing };
	ImGui::SetNextWindowPos(optionPos);
	ImGui::SetNextWindowSize(optionSize);
	if (ImGui::Begin("Option", nullptr, wf))
	{
		SetCameraGUI(row, camera);
		SetRenderingGUI(row);

		SetModelsGUI(row);

		if (pass_manager_.cpu_or_gpu_ == vku::CpuOrGpu::GPU)
			SetSimulationGUI(row, targetSimFPS, simDt, paused, pauseEachframe);

	}
	ImGui::End();

	// Right Side
	ImVec2 statGuiPos{ swapchain_.swapchain_extent_.width - swapchain_.swapchain_extent_.width * 0.25f - spacing, spacing };
	ImVec2 StatGuiSize{ swapchain_.swapchain_extent_.width * 0.25f, 0 };
	ImGui::SetNextWindowPos(statGuiPos);
	ImGui::SetNextWindowSize(StatGuiSize);
	if (ImGui::Begin("Performance Monitor", nullptr, wf))
	{
		SetStatGUI(row);

		SetTimeingGUI(row);
	}
	ImGui::End();

	//if (ImGui::Begin("ShadowBuffer", nullptr))
	//{
	//	auto& id = imgui_id_[imgui_id_.size() - 1];
	//	ImGui::ImageButton(id.first.c_str(), id.second, ImVec2(128, 128));
	//}
	//ImGui::End();

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

template<typename RowFn>
void GUI::SetModelsGUI(RowFn&& row)
{

	auto row3 = [&](const char* label, auto&& drawControl, auto&& drawExtra)
		{
			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(label);

			ImGui::TableSetColumnIndex(1);
			ImGui::SetNextItemWidth(-FLT_MIN);
			drawControl();

			ImGui::TableSetColumnIndex(2);
			ImGui::SetNextItemWidth(-FLT_MIN);
			drawExtra();
		};


	auto SelectPopUp = [&](const char* label, int& idx)
		{
			if (ImGui::Button(label))
				ImGui::OpenPopup(label);
			ImGui::SetNextWindowSize(ImVec2(520, 360), ImGuiCond_Appearing);
			if (ImGui::BeginPopup(label))
			{
				const int cols = 6;
				const float thumb = 64.0f;

				ImGui::BeginChild("tex_scroll", ImVec2(0, 0), true);

				if (ImGui::BeginTable("tex_grid", cols))
				{
					for (int i = 0; i < (int)imgui_id_.size(); ++i)
					{
						ImGui::TableNextColumn();

						auto& id = imgui_id_[i];
						if (ImGui::ImageButton(id.first.c_str(), id.second, ImVec2(thumb, thumb)))
						{
							idx = i;
							ImGui::CloseCurrentPopup();
						}
						//ImGui::TextWrapped("%s", id.first.c_str());
					}
					ImGui::EndTable();
				}

				ImGui::EndChild();
				ImGui::EndPopup();
			}
		};

	auto CheckEnable = [&](const char* label, uint32_t& enable) {
		bool check = enable; ImGui::Checkbox(label, &check); enable = check;
		};

	auto& models = model_manager_.models_;
	auto& clothes = pass_manager_.particle_manager_->clothes_;
	auto& softbodies = pass_manager_.particle_manager_->softbodies_;

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen))
	{
		for (auto& model : models)
		{
			auto& ubo = model->ubo_data;

			if (ImGui::TreeNode(model->name_.c_str()))
			{
				{
					ImGui::SeparatorText("Factor");
					ImGui::Indent();
					ImGui::BeginChild("Factor", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

					if (ImGui::BeginTable("Factor", 2,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{

						row("Albedo", [&] { ImGui::DragFloat3("##Albedo", &ubo.albedo[0], 0.1f, 0.0f, 1.0f); });
						row("AO", [&] { ImGui::DragFloat("##AO", &ubo.ao_factor, 0.1f, 0.0f, 1.0f); });
						row("Roughness", [&] { ImGui::DragFloat("##Roughness", &ubo.roughness_factor, 0.1f, 0.0f, 1.0f); });
						row("Meltallic", [&] { ImGui::DragFloat("##Meltallic", &ubo.metallic_factor, 0.1f, 0.0f, 1.0f); });
						row("Height", [&] { ImGui::DragFloat("##Height", &ubo.height_factor, 0.001f, 0.0f, 1.0f); });
						row("Coat", [&] { ImGui::DragFloat("##Coat", &ubo.coat_factor, 0.001f, 0.0f, 1.0f); });
						row("CoatRoughness", [&] { ImGui::DragFloat("##CoatRoughness", &ubo.coat_roughness_factor, 0.001f, 0.0f, 1.0f); });
						row("Fuzz", [&] { ImGui::DragFloat("##Fuzz", &ubo.fuzz_factor, 0.001f, 0.0f, 1.0f); });
						row("FuzzRoughness", [&] { ImGui::DragFloat("##FuzzRoughness", &ubo.fuzz_roughness_factor, 0.001f, 0.0f, 1.0f); });
						ImGui::EndTable();
					}
					ImGui::EndChild();
					ImGui::Unindent();

				}

				{
					ImGui::SeparatorText("Enable");
					ImGui::Indent();
					ImGui::BeginChild("Enable", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

					if (ImGui::BeginTable("Enable", 3,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{
						row3("Albedo",
							[&] { CheckEnable("##Albedo", ubo.albedo_enable); },
							[&] { SelectPopUp("Select##Albedo", ubo.albedo_idx); });
						row3("Normal",
							[&] { CheckEnable("##Normal", ubo.normal_enable); },
							[&] { SelectPopUp("Select##Normal", ubo.normal_idx); });
						row3("ARM",
							[&] { CheckEnable("##ARM", ubo.arm_enable); },
							[&] { SelectPopUp("Select##ARM", ubo.arm_idx); });
						row3("AO",
							[&] { CheckEnable("##AO", ubo.ao_enable); },
							[&] { SelectPopUp("Select##AO", ubo.ao_idx); });
						row3("Roughtness",
							[&] { CheckEnable("##Roughness", ubo.roughness_enable); },
							[&] { SelectPopUp("Select##Roughness", ubo.roughness_idx); });
						row3("Meltallic",
							[&] { CheckEnable("##Meltallic", ubo.metallic_enable); },
							[&] { SelectPopUp("Select##Meltallic", ubo.metallic_idx); });
						row3("Height",
							[&] { CheckEnable("##Height", ubo.height_enable); },
							[&] { SelectPopUp("Select##Height", ubo.height_idx); });
						row("CheckerBoard", [&] { CheckEnable("##CheckerBoard", ubo.checker_board_enable); });
						row("Movable", [&] { ImGui::Checkbox("##Movable", &model->movable_); });
						row("Render", [&] { ImGui::Checkbox("##Render", &model->render_); });

						row("ShapeRender", [&] { ImGui::Checkbox("##ShapeRender", &model->shape_collision_render_); });
						row("ShapeCollide", [&] { model->shape_collision_update_ = ImGui::Checkbox("##ShapeCollide", &model->shape_collision_collide_); });

						if (model->model_type_ == ModelType::SKINNED)
						{
							row("Animation", [&] { ImGui::Checkbox("##Animation", &model->do_animation); });
							row("CapsuleRender", [&] { ImGui::Checkbox("##CapsuleRender", &model->capsule_collision_render_); });
							row("CapsuleCollide", [&] { model->capsule_collision_update_ = ImGui::Checkbox("##CapsuleCollide", &model->capsule_collision_collide_); });
						}
						ImGui::EndTable();
					}

					ImGui::EndChild();
					ImGui::Unindent();
				}

				ImGui::TreePop();
			}
		}

		ImGui::SeparatorText("Cloth");
		ImGui::Indent();
		ImGui::BeginChild("Cloth", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
		for (auto& cloth : clothes)
		{
			if (ImGui::TreeNode(cloth.name.c_str()))
			{
				{
					ImGui::SeparatorText("Factor");
					ImGui::Indent();
					ImGui::BeginChild("Factor", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
					if (ImGui::BeginTable("Factor", 2,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{
						row("Albedo", [&] { ImGui::DragFloat3("##Albedo", &cloth.ubo_data.albedo[0], 0.1f, 0.0f, 1.0f); });
						row("AO", [&] { ImGui::DragFloat("##AO", &cloth.ubo_data.ao_factor, 0.1f, 0.0f, 1.0f); });
						row("Roughness", [&] { ImGui::DragFloat("##Roughness", &cloth.ubo_data.roughness_factor, 0.1f, 0.0f, 1.0f); });
						row("Meltallic", [&] { ImGui::DragFloat("##Meltallic", &cloth.ubo_data.metallic_factor, 0.1f, 0.0f, 1.0f); });
						row("Height", [&] { ImGui::DragFloat("##Height", &cloth.ubo_data.height_factor, 0.001f, 0.0f, 1.0f); });
						row("Coat", [&] { ImGui::DragFloat("##Coat", &cloth.ubo_data.coat_factor, 0.001f, 0.0f, 1.0f); });
						row("CoatRoughness", [&] { ImGui::DragFloat("##CoatRoughness", &cloth.ubo_data.coat_roughness_factor, 0.001f, 0.0f, 1.0f); });
						row("Fuzz", [&] { ImGui::DragFloat("##Fuzz", &cloth.ubo_data.fuzz_factor, 0.001f, 0.0f, 1.0f); });
						row("FuzzRoughness", [&] { ImGui::DragFloat("##FuzzRoughness", &cloth.ubo_data.fuzz_roughness_factor, 0.001f, 0.0f, 1.0f); });
						row("TileUV", [&] { ImGui::DragFloat2("##TileUV", &cloth.ubo_data.tile_uv[0], 1.0f, 0.0f, 100.0f); });

						ImGui::EndTable();
					}
					ImGui::EndChild();
					ImGui::Unindent();
				}


				{
					ImGui::SeparatorText("Enable");
					ImGui::Indent();
					ImGui::BeginChild("Enable", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
					if (ImGui::BeginTable("Enable", 3,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{
						row3("Albedo",
							[&] { CheckEnable("##Albedo", cloth.ubo_data.albedo_enable); },
							[&] { SelectPopUp("Select##Albedo", cloth.ubo_data.albedo_idx); });
						row3("Normal",
							[&] { CheckEnable("##Normal", cloth.ubo_data.normal_enable); },
							[&] { SelectPopUp("Select##Normal", cloth.ubo_data.normal_idx); });
						row3("ARM",
							[&] { CheckEnable("##ARM", cloth.ubo_data.arm_enable); },
							[&] { SelectPopUp("Select##ARM", cloth.ubo_data.arm_idx); });
						row3("AO",
							[&] { CheckEnable("##AO", cloth.ubo_data.ao_enable); },
							[&] { SelectPopUp("Select##AO", cloth.ubo_data.ao_idx); });
						row3("Roughtness",
							[&] { CheckEnable("##Roughtness", cloth.ubo_data.roughness_enable); },
							[&] { SelectPopUp("Select##Roughtness", cloth.ubo_data.roughness_idx); });
						row3("Meltallic",
							[&] { CheckEnable("##Meltallic", cloth.ubo_data.metallic_enable); },
							[&] { SelectPopUp("Select##Meltallic", cloth.ubo_data.metallic_idx); });
						row3("Height",
							[&] { CheckEnable("##Height", cloth.ubo_data.height_enable); },
							[&] { SelectPopUp("Select##Height", cloth.ubo_data.height_idx); });
						ImGui::EndTable();
					}
					ImGui::EndChild();
					ImGui::Unindent();
				}

				ImGui::TreePop();
			}
		}
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("Softbody");
		ImGui::Indent();
		ImGui::BeginChild("Softbody", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		for (auto& softbody : softbodies)
		{
			if (ImGui::TreeNode(softbody.name.c_str()))
			{
				{
					ImGui::SeparatorText("Factor");
					ImGui::Indent();
					ImGui::BeginChild("Factor", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
					if (ImGui::BeginTable("Factor", 2,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{
						row("Albedo", [&] { ImGui::DragFloat3("##Albedo", &softbody.ubo_data.albedo[0], 0.1f, 0.0f, 1.0f); });
						row("AO", [&] { ImGui::DragFloat("##AO", &softbody.ubo_data.ao_factor, 0.1f, 0.0f, 1.0f); });
						row("Roughness", [&] { ImGui::DragFloat("##Roughness", &softbody.ubo_data.roughness_factor, 0.1f, 0.0f, 1.0f); });
						row("Meltallic", [&] { ImGui::DragFloat("##Meltallic", &softbody.ubo_data.metallic_factor, 0.1f, 0.0f, 1.0f); });
						row("Height", [&] { ImGui::DragFloat("##Height", &softbody.ubo_data.height_factor, 0.001f, 0.0f, 1.0f); });
						row("Coat", [&] { ImGui::DragFloat("##Coat", &softbody.ubo_data.coat_factor, 0.001f, 0.0f, 1.0f); });
						row("CoatRoughness", [&] { ImGui::DragFloat("##CoatRoughness", &softbody.ubo_data.coat_roughness_factor, 0.001f, 0.0f, 1.0f); });
						row("Fuzz", [&] { ImGui::DragFloat("##Fuzz", &softbody.ubo_data.fuzz_factor, 0.001f, 0.0f, 1.0f); });
						row("FuzzRoughness", [&] { ImGui::DragFloat("##FuzzRoughness", &softbody.ubo_data.fuzz_roughness_factor, 0.001f, 0.0f, 1.0f); });
						row("TileUV", [&] { ImGui::DragFloat2("##TileUV", &softbody.ubo_data.tile_uv[0], 1.0f, 0.0f, 100.0f); });

						ImGui::EndTable();
					}
					ImGui::EndChild();
					ImGui::Unindent();
				}

				auto CheckEnable = [&](const char* label, uint32_t& enable) {
					bool check = enable; ImGui::Checkbox(label, &check); enable = check;
					};

				{
					ImGui::SeparatorText("Enable");
					ImGui::Indent();
					ImGui::BeginChild("Enable", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
					if (ImGui::BeginTable("Enable", 3,
						ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
					{
						row3("Albedo",
							[&] { CheckEnable("##Albedo", softbody.ubo_data.albedo_enable); },
							[&] { SelectPopUp("Select##Albedo", softbody.ubo_data.albedo_idx); });
						row3("Normal",
							[&] { CheckEnable("##Normal", softbody.ubo_data.normal_enable); },
							[&] { SelectPopUp("Select##Normal", softbody.ubo_data.normal_idx); });
						row3("ARM",
							[&] { CheckEnable("##ARM", softbody.ubo_data.arm_enable); },
							[&] { SelectPopUp("Select##ARM", softbody.ubo_data.arm_idx); });
						row3("AO",
							[&] { CheckEnable("##AO", softbody.ubo_data.ao_enable); },
							[&] { SelectPopUp("Select##AO", softbody.ubo_data.ao_idx); });
						row3("Roughtness",
							[&] { CheckEnable("##Roughtness", softbody.ubo_data.roughness_enable); },
							[&] { SelectPopUp("Select##Roughtness", softbody.ubo_data.roughness_idx); });
						row3("Meltallic",
							[&] { CheckEnable("##Meltallic", softbody.ubo_data.metallic_enable); },
							[&] { SelectPopUp("Select##Meltallic", softbody.ubo_data.metallic_idx); });
						row3("Height",
							[&] { CheckEnable("##Height", softbody.ubo_data.height_enable); },
							[&] { SelectPopUp("Select##Height", softbody.ubo_data.height_idx); });
						ImGui::EndTable();
					}
					ImGui::EndChild();
					ImGui::Unindent();
				}

				ImGui::TreePop();
			}

		}
		ImGui::EndChild();
		ImGui::Unindent();
	}

}

template<typename RowFn>
void GUI::SetTimeingGUI(RowFn&& row)
{
	auto& sim = *pass_manager_.sim_pass_gpu_;

	auto& labels = sim.labels_;
	auto& labelToTime = sim.label_time_;
	auto& labelToAvgTime = sim.label_avg_time_;

	open_timestamps_ = ImGui::CollapsingHeader("Timing"); //, ImGuiTreeNodeFlags_DefaultOpen
	if (open_timestamps_)
	{
		if (ImGui::BeginTable("Timing", 4))//,  ImGuiTableFlags_BordersOuter))
		{
			ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch, 1.0f);
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
		ImGui::SeparatorText("GpuTime");
		ImGui::Text("Compute : %.3f", sim.pass_total_time_);
		ImGui::Text("Graphics : %.3f", pass_manager_.graphics_pass_->pass_total_time_);

		count_ = sim.time_count_;
	}
}

template<typename RowFn>
void GUI::SetSimulationGUI(RowFn&& row, float& targetSimFPS, double& simDt, bool& paused, bool& pauseEachframe)
{
	auto& sim = *pass_manager_.sim_pass_gpu_;

	if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		auto& pm = *pass_manager_.particle_manager_;

		ImGui::SeparatorText("Parameter");

		ImGui::Indent();
		ImGui::BeginChild("Parameter", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		if (ImGui::BeginTable("Parameter", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Pause(X)", [&] { ImGui::Checkbox("##Pause", &paused); });
			row("PauseEachframe(Z,X)", [&] { ImGui::Checkbox("##PauseEachframe", &pauseEachframe); });
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
			row("CollideCompliance", [&] { ImGui::DragFloat("##CollideCompliance", &sim.datas_.compliance.collide,
				1e-9f, 0.0f, 1.0f, "%.9f"); });
			row("NeighborFriction", [&] { ImGui::DragFloat("##NeighborFriction", &sim.ubo_.datas.sim_params.neighbor_friction, 0.1f, 0.0f, 10.0f, "%.1f"); });
			row("MaxNeighbors", [&] { int maxNeighbors = sim.ubo_.datas.sim_params.max_neighbors;  ImGui::DragInt("##MaxNeighbors", &maxNeighbors, 1, 0, 20); sim.ubo_.datas.sim_params.max_neighbors = maxNeighbors; });
			row("MaxSpeed", [&] { ImGui::DragFloat("##MaxSpeed", &sim.ubo_.datas.sim_params.max_speed, 0.1f, sim.ubo_.datas.sim_params.max_speed, sim.ubo_.datas.sim_params.max_speed, "%.1f"); });
			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("SimulationFactors");
		if (ImGui::TreeNode("Cloth"))
		{
			ImGui::SeparatorText("SolverConfig");
			ImGui::Indent();
			ImGui::BeginChild("SolverConfig", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
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
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::SeparatorText("Stiffness");
			ImGui::Indent();
			ImGui::BeginChild("Stiffness", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("Stiffness", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::DragFloat("##StretchStiffness", &sim.ubo_.datas.sim_params.stretch_stiffness, 1e-2f, 0.0f, 100.0f, "%.3f"); });
				row("Shear", [&] { ImGui::DragFloat("##ShearStiffness", &sim.ubo_.datas.sim_params.shear_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
				row("Bend", [&] { ImGui::DragFloat("##BendStiffness", &sim.ubo_.datas.sim_params.bend_stiffness, 1e-3f, 0.0f, 2.0f, "%.3f"); });
				row("Area", [&] { ImGui::DragFloat("##AreaStiffness", &sim.ubo_.datas.sim_params.area_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
				row("SelfCollision", [&] { ImGui::DragFloat("##SelfCollision", &sim.ubo_.datas.sim_params.self_collision_stiffness, 1.0f, 0.0f, 100.0f, "%.1f"); });
				ImGui::EndTable();
			}
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::SeparatorText("Compliance");
			ImGui::Indent();
			ImGui::BeginChild("Compliance", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("Compliance", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim.datas_.compliance.stretch,
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
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::SeparatorText("Beta");
			ImGui::Indent();
			ImGui::BeginChild("Beta", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("Beta", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::DragFloat("##Stretch", &sim.datas_.beta.stretch, 1.0f, 0.0f, 1000.0f, "%.1f"); });

				ImGui::EndTable();
			}
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Softbody"))
		{
			ImGui::SeparatorText("SolverConfig");
			ImGui::Indent();
			ImGui::BeginChild("SolverConfig", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("SolverConfig", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::Checkbox("##SoftbodyStretch", &sim.solver_config_.softbody_stretch); });
				row("Volume", [&] { ImGui::Checkbox("##SoftbodyVolume", &sim.solver_config_.softbody_volume); });
				ImGui::EndTable();
			}
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::SeparatorText("Stiffness");
			ImGui::Indent();
			ImGui::BeginChild("Stiffness", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("Stiffness", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::DragFloat("##SoftbodyStretch", &sim.ubo_.datas.sim_params.softbody_stretch_stiffness, 1e-2f, 0.0f, 100.0f, "%.1f"); });
				row("Volume", [&] { ImGui::DragFloat("##SoftbodyVolume", &sim.ubo_.datas.sim_params.volume_stiffness, 1e-2f, 0.0f, 100.0f, "%.1f"); });
				ImGui::EndTable();
			}
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::SeparatorText("Compliance");
			ImGui::Indent();
			ImGui::BeginChild("Compliance", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
			if (ImGui::BeginTable("Compliance", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("Stretch", [&] { ImGui::DragFloat("##SoftbodyStretch", &sim.datas_.compliance.softbody_stretch,
					1e-9f, 0.0f, 1.0f, "%.9f"); });
				row("Volume", [&] { ImGui::DragFloat("##SoftbodyVolume", &sim.datas_.compliance.softbody_volume,
					1e-9f, 0.0f, 1.0f, "%.9f"); });

				ImGui::EndTable();
			}
			ImGui::EndChild();
			ImGui::Unindent();

			ImGui::TreePop();
		}

	}
}

template<typename RowFn>
void GUI::SetTestSceneGUI(RowFn&& row)
{
	auto& scene = pass_manager_.test_scene_;

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

		auto CheckEnable = [&](const char* label, uint32_t& enable) {
			bool check = enable; ImGui::Checkbox(label, &check); enable = check;
			};

		ImGui::SeparatorText("PolygonMode");
		ImGui::Indent();
		ImGui::BeginChild("PolygonMode", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		const char* items[] = { "Solid", "Wireframe", "Point" };
		int item_current = gp.polygon_mode_;
		ImGui::ListBox("##", &item_current, items, IM_ARRAYSIZE(items), 3);
		gp.polygon_mode_ = vku::PolygonMode(item_current);
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("SpotLight");
		ImGui::Indent();
		ImGui::BeginChild("SpotLight", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		if (ImGui::BeginTable("SpotLight", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { CheckEnable("##Enable", gp.ubo_datas_.light.light_enable); });
			row("Pos", [&] { ImGui::DragFloat3("##Pos", &gp.ubo_datas_.light.position[0], 0.1f); });
			row("Dir", [&] { ImGui::DragFloat3("##Dir", &gp.ubo_datas_.light.direction[0], 0.1f); });
			row("Inner", [&] { ImGui::DragFloat("##Inner", &gp.ubo_datas_.light.inner, 0.1f); });
			row("Outer", [&] { ImGui::DragFloat("##Outer", &gp.ubo_datas_.light.outer, 0.1f); });
			row("Intensity", [&] { ImGui::DragFloat("##Intensity", &gp.ubo_datas_.light.intensity, 0.1f, 0.0f, 10000.0f); });

			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("Shadow");
		ImGui::Indent();
		ImGui::BeginChild("Shadow", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		if (ImGui::BeginTable("Shadow", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { CheckEnable("##Enable", gp.ubo_datas_.light.shadow_enable); });
			row("Strength", [&] { ImGui::DragFloat("##Strength", &gp.ubo_datas_.light.shadow_strength, 0.1f, 0.0f, 1.0f); });

			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("PBR");
		ImGui::Indent();
		ImGui::BeginChild("PBR", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
		if (ImGui::BeginTable("PBR", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Enable", [&] { CheckEnable("##Enable", gp.ubo_datas_.light.pbr_enable); });
			row("Exposure", [&] { ImGui::DragFloat("##Exposure", &gp.ubo_datas_.light.exposure, 0.1f, 0.0f, 2.0f); });
			ImGui::EndTable();
		}
		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("Skybox");

		ImGui::Indent();
		ImGui::BeginChild("Skybox", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
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

		ImGui::EndChild();
		ImGui::Unindent();
	}
}

template<typename RowFn>
void GUI::SetStatGUI(RowFn&& row)
{
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

	auto& pm = *pass_manager_.particle_manager_;
	auto& d = pass_manager_.sim_pass_gpu_->datas_;

	if (ImGui::CollapsingHeader("Stat"))
	{

		ImGui::SeparatorText("Cloth");
		ImGui::Indent();
		ImGui::BeginChild("Cloth", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
		if (ImGui::BeginTable("Stat", 2, ImGuiTableFlags_BordersInnerV))
		{
			row("NumParticles", [&] { ImGui::Text("%u", pm.num_cloth_particles_); });
			row("NumEdges", [&] { ImGui::Text("%u", d.num_cloth_edges); });
			row("NumShears", [&] { ImGui::Text("%u", d.num_shears); });
			row("NumBends", [&] { ImGui::Text("%u", d.num_bends); });
			row("NumAreas", [&] { ImGui::Text("%u", d.num_areas); });

			ImGui::EndTable();
		}

		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("Softbody");
		ImGui::Indent();
		ImGui::BeginChild("Softbody", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
		if (ImGui::BeginTable("Stat", 2, ImGuiTableFlags_BordersInnerV))
		{
			row("NumParticles", [&] { ImGui::Text("%u", pm.num_softbody_particles_); });
			row("NumEdges", [&] { ImGui::Text("%u", d.num_softbody_edges); });
			row("NumVolumes", [&] { ImGui::Text("%u", d.num_volumes); });

			ImGui::EndTable();
		}

		ImGui::EndChild();
		ImGui::Unindent();

		ImGui::SeparatorText("Total");
		ImGui::Indent();
		ImGui::BeginChild("Total", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);
		if (ImGui::BeginTable("Stat", 2, ImGuiTableFlags_BordersInnerV))
		{
			row("NumParticles", [&] { ImGui::Text("%u", pm.total_particles_); });
			row("NumEdges", [&] { ImGui::Text("%u", d.num_edges); });
			row("NumShears", [&] { ImGui::Text("%u", d.num_shears); });
			row("NumBends", [&] { ImGui::Text("%u", d.num_bends); });
			row("NumAreas", [&] { ImGui::Text("%u", d.num_areas); });
			row("NumVolumes", [&] { ImGui::Text("%u", d.num_volumes); });

			ImGui::EndTable();
		}

		ImGui::EndChild();
		ImGui::Unindent();
	}
}

template<typename RowFn>
void GUI::SetCameraGUI(RowFn&& row, Camera& camera)
{
	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		ImGui::SeparatorText("Move");

		ImGui::Indent();
		ImGui::BeginChild("Move", ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

		if (ImGui::BeginTable("Camera", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
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
		ImGui::EndChild();
		ImGui::Unindent();
	}
}