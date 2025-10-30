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

	void Update(const Camera& camera, const glm::vec2& viewportSize, Model& model);

	bool is_left_button_down_event = false;
	bool is_left_button_up_event = false;
	bool is_right_button_down_event = false;
	bool is_right_button_up_event = false;

	glm::vec2 mouse_pos_{0.0f, 0.0f};

private:
	Ray CalculateMouseRay(const Camera& camera, const glm::vec2& viewportSize);
	// 우클릭 방식이 near/far를 쓰므로 보조 함수 하나 더:
	void CalculateMouseNearFar(const Camera& camera, const glm::vec2& vp, glm::vec3& outNear, glm::vec3& outFar);


	// 회전 상태
	bool is_dragging_ = false;
	bool has_prev_ = false;
	glm::vec3 prevVector_{ 0.0f };

	// 우클릭 이동 상태
	bool is_translating_ = false;
	bool has_grab_point_ = false;
	float prevRatio_ = 0.0f;   // dist / |far-near|
	glm::vec3 prevPos_{ 0.0f };      // 지난 프레임의 월드 교점
};

