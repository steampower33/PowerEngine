#include "camera.h"
#include "model.h"
#include "ray.h"

#include "mouse_interactor.h"

MouseInteractor::MouseInteractor()
{

}

std::pair<int, float> MouseInteractor::PickClosestModel(
    const Ray& ray, const std::vector<std::unique_ptr<Model>>& models) const
{
    int picked = -1;
    float minDist = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(models.size()); ++i) {
        float dist = 0.0f;
        if (models[i]->movable_ && ray.Intersects(*models[i], dist)) {
            if (dist < minDist) {
                minDist = dist;
                picked = i;
            }
        }
    }
    return { picked, minDist };
}

void MouseInteractor::Update(const Camera& camera,
    const glm::vec2& viewportSize,
    std::vector<std::unique_ptr<Model>>& models)
{
    const float EPS = 1e-6f;

    if (selected_ >= static_cast<int>(models.size()))
        selected_ = -1;

    if (is_left_button_down_event && !is_dragging_ && !is_translating_) {
        Ray ray = CalculateMouseRay(camera, viewportSize);
        auto [idx, dist] = PickClosestModel(ray, models);

        if (idx >= 0) {
            selected_ = idx;
            is_dragging_ = true;
            has_prev_ = true;

            const Model& m = *models[selected_];
            glm::vec3 center = m.position_;
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
        else {
            selected_ = -1;
        }
    }

    if (is_right_button_down_event && !is_translating_ && !is_dragging_) {
        Ray ray = CalculateMouseRay(camera, viewportSize);
        auto [idx, dist] = PickClosestModel(ray, models);

        if (idx >= 0) {
            selected_ = idx;

            glm::vec3 wNear, wFar;
            CalculateMouseNearFar(camera, viewportSize, wNear, wFar);
            glm::vec3 nf = wFar - wNear;
            float nfLen = glm::length(nf);
            if (nfLen > EPS) {
                is_translating_ = true;
                has_grab_point_ = true;

                prevRatio_ = dist / nfLen;
                prevPos_ = wNear + nf * prevRatio_;
            }
        }
        else {
            selected_ = -1;
        }
    }

    if (is_dragging_ && has_prev_ && selected_ >= 0) {
        Ray ray = CalculateMouseRay(camera, viewportSize);
        float dist = 0.0f;
        Model& model = *models[selected_];

        if (model.movable_ && ray.Intersects(model, dist)) {
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
                        model.ApplyTransform(dq, glm::vec3(0.0f)); // È¸Àü
                    }
                }
                prevVector_ = curr;
            }
        }
    }

    if (is_translating_ && has_grab_point_ && selected_ >= 0) {
        glm::vec3 wNear, wFar;
        CalculateMouseNearFar(camera, viewportSize, wNear, wFar);
        glm::vec3 nf = wFar - wNear;
        float nfLen = glm::length(nf);
        if (nfLen > EPS) {
            glm::vec3 newPos = wNear + nf * prevRatio_;
            glm::vec3 delta = newPos - prevPos_;

            if (glm::length(delta) > 1e-8f) {
                models[selected_]->ApplyTransform(glm::quat(1, 0, 0, 0), delta);
                prevPos_ = newPos;
            }
        }
    }

    if (is_left_button_up_event) {
        is_dragging_ = false;
        has_prev_ = false;
    }
    if (is_right_button_up_event) {
        is_translating_ = false;
        has_grab_point_ = false;
        prevRatio_ = 0.0f;
    }

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
