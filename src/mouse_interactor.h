#pragma once

struct Camera;
struct Ray;

class MouseInteractor
{
public:
	MouseInteractor();
	MouseInteractor(const MouseInteractor& rhs) = delete;
	MouseInteractor(MouseInteractor&& rhs) = delete;
	MouseInteractor& operator=(const MouseInteractor& rhs) = delete;
	MouseInteractor& operator=(MouseInteractor&& rhs) = delete;
	~MouseInteractor() = default;

	void Update(const Camera& camera, const glm::vec2& viewportSize, std::vector<std::unique_ptr<Model>>& models);

	bool is_left_button_down_event = false;
	bool is_left_button_up_event = false;
	bool is_right_button_down_event = false;
	bool is_right_button_up_event = false;

	glm::vec2 mouse_pos_{0.0f, 0.0f};

private:
	Ray CalculateMouseRay(const Camera& camera, const glm::vec2& viewportSize);
	void CalculateMouseNearFar(const Camera& camera, const glm::vec2& vp, glm::vec3& outNear, glm::vec3& outFar);

	bool is_dragging_ = false;
	bool is_translating_ = false;

	bool has_prev_ = false;
	glm::vec3 prevVector_{ 0.0f };

	bool has_grab_point_ = false;
	float prevRatio_ = 0.0f;
	glm::vec3 prevPos_{ 0.0f };

	int selected_ = -1;

	std::pair<int, float> PickClosestModel(const Ray& ray,
		const std::vector<std::unique_ptr<Model>>& models) const;
};
