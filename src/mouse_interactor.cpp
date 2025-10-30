#include "camera.h"
#include "model.h"
#include "ray.h"

#include "mouse_interactor.h"

MouseInteractor::MouseInteractor()
{

}

void MouseInteractor::Update(const Camera& camera, const glm::vec2& viewportSize, Model& model)
{
    const float EPS = 1e-6f;

    // ===== 좌클릭: 회전 시작 =====
    if (is_left_button_down_event && !is_dragging_ && !is_translating_) {
        Ray ray = CalculateMouseRay(camera, viewportSize);
        float dist = 0.0f;
        if (ray.Intersects(model, dist)) {
            is_dragging_ = true;
            has_prev_ = true;

            glm::vec3 center = model.position_;
            glm::vec3 pickPoint = ray.origin + ray.direction * dist;
            glm::vec3 v = pickPoint - center;
            float len = glm::length(v);
            if (len < EPS) {
                has_prev_ = false;
                is_dragging_ = false;
            }
            else {
                prevVector_ = v / len;
            }
        }
    }

    // ===== 우클릭: 이동 시작 (깊이 비율 고정) =====
    if (is_right_button_down_event && !is_translating_ && !is_dragging_) {
        // 먼저 피킹해서 dist 얻기
        Ray ray = CalculateMouseRay(camera, viewportSize);
        float dist = 0.0f;
        if (ray.Intersects(model, dist)) {
            // near/far로 비율 계산
            glm::vec3 wNear, wFar;
            CalculateMouseNearFar(camera, viewportSize, wNear, wFar);
            glm::vec3 nf = wFar - wNear;
            float nfLen = glm::length(nf);
            if (nfLen > EPS) {
                is_translating_ = true;
                has_grab_point_ = true;

                prevRatio_ = dist / nfLen;                      // DX12 코드의 nearToFar 비율
                prevPos_ = wNear + nf * prevRatio_;           // 시작 교점 저장
            }
        }
    }

    // ===== 좌 드래그: 회전 진행 =====
    if (is_dragging_ && has_prev_) {
        Ray ray = CalculateMouseRay(camera, viewportSize);
        float dist = 0.0f;
        if (ray.Intersects(model, dist)) {
            glm::vec3 center = model.position_;
            glm::vec3 pickPoint = ray.origin + ray.direction * dist;
            glm::vec3 v = pickPoint - center;
            float len = glm::length(v);
            if (len >= EPS) {
                glm::vec3 curr = v / len;

                float d = glm::clamp(glm::dot(prevVector_, curr), -1.0f, 1.0f);
                float angle = acosf(d);
                if (angle > 1e-4f) {
                    glm::vec3 axis;
                    if (d < -0.9999f) {
                        glm::vec3 ortho = (glm::abs(prevVector_.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
                        axis = glm::cross(prevVector_, ortho);
                        float al = glm::length(axis);
                        if (al < EPS) { ortho = glm::vec3(0, 0, 1); axis = glm::cross(prevVector_, ortho); al = glm::length(axis); }
                        if (al >= EPS) axis /= al; else angle = 0.0f;
                    }
                    else {
                        axis = glm::cross(prevVector_, curr);
                        float al = glm::length(axis);
                        if (al >= EPS) axis /= al; else angle = 0.0f;
                    }
                    if (angle > 0.0f) {
                        glm::quat dq = glm::angleAxis(angle, axis);
                        model.ApplyTransform(dq, glm::vec3(0.0f)); // 회전만
                    }
                }
                prevVector_ = curr;
            }
        }
    }

    // ===== 우 드래그: 이동 진행 (깊이 비율 유지) =====
    if (is_translating_ && has_grab_point_) {
        glm::vec3 wNear, wFar;
        CalculateMouseNearFar(camera, viewportSize, wNear, wFar);
        glm::vec3 nf = wFar - wNear;
        float nfLen = glm::length(nf);
        if (nfLen > EPS) {
            glm::vec3 newPos = wNear + nf * prevRatio_;
            glm::vec3 delta = newPos - prevPos_;

            // 너무 작은 이동은 무시 (디바운스)
            if (glm::length(delta) > 1e-8f) {
                model.ApplyTransform(glm::quat(1, 0, 0, 0), delta);
                prevPos_ = newPos;
            }
        }
    }

    // ===== 버튼 업 =====
    if (is_left_button_up_event) {
        is_dragging_ = false;
        has_prev_ = false;
    }
    if (is_right_button_up_event) {
        is_translating_ = false;
        has_grab_point_ = false;
        prevRatio_ = 0.0f;
    }

    // 이벤트 플래그 리셋
    is_left_button_down_event = false;
    is_left_button_up_event = false;
    is_right_button_down_event = false;
    is_right_button_up_event = false;
}

Ray MouseInteractor::CalculateMouseRay(const Camera& camera, const glm::vec2& viewportSize)
{
    const float ndcX = (2.0f * mouse_pos_.x) / viewportSize.x - 1.0f;
    const float ndcY = (2.0f * mouse_pos_.y) / viewportSize.y - 1.0f;

    glm::vec4 rayClipNear(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 rayClipFar(ndcX, ndcY, 1.0f, 1.0f);

    const glm::mat4 invProjection = glm::inverse(camera.Proj(viewportSize.x, viewportSize.y));
    glm::vec4 rayViewNear = invProjection * rayClipNear;
    glm::vec4 rayViewFar = invProjection * rayClipFar;

    rayViewNear /= rayViewNear.w;
    rayViewFar /= rayViewFar.w;

    const glm::mat4 invView = glm::inverse(camera.View());
    glm::vec4 rayWorldNear_w = invView * rayViewNear;
    glm::vec4 rayWorldFar_w = invView * rayViewFar;

    glm::vec3 rayWorldNear(rayWorldNear_w);
    glm::vec3 rayWorldFar(rayWorldFar_w);

    glm::vec3 direction = glm::normalize(rayWorldFar - rayWorldNear);

    return Ray(rayWorldNear, direction);
}

void MouseInteractor::CalculateMouseNearFar(const Camera& camera, const glm::vec2& viewportSize,
    glm::vec3& outNear, glm::vec3& outFar)
{
    const float ndcX = (2.0f * mouse_pos_.x) / viewportSize.x - 1.0f;
    const float ndcY = (2.0f * mouse_pos_.y) / viewportSize.y - 1.0f;

    glm::vec4 rayClipNear(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 rayClipFar(ndcX, ndcY, 1.0f, 1.0f);

    const glm::mat4 invP = glm::inverse(camera.Proj(viewportSize.x, viewportSize.y));
    glm::vec4 rayViewNear = invP * rayClipNear;
    glm::vec4 rayViewFar = invP * rayClipFar;
    rayViewNear /= rayViewNear.w;
    rayViewFar /= rayViewFar.w;

    const glm::mat4 invV = glm::inverse(camera.View());
    glm::vec4 wNear = invV * rayViewNear;
    glm::vec4 wFar = invV * rayViewFar;

    outNear = glm::vec3(wNear);
    outFar = glm::vec3(wFar);
}