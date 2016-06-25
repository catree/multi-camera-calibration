#include "calibration.h"
#include <opencv\cv.h>
#include <Eigen\Dense>

const int Z_THRESH = 2000;
const int SCORE_DIVIDER = 5;
const float REPROJ_CONV = 10;
const float REPROJ_STEP = 2.0f;
const int ITER_MAX = 25;

float CalibrationMethod::computeReprojectionError(float3 target, float fx, float fy, float ppx, float ppy, float3 source, float4x4 & pose)
{
    using linalg::aliases::float4;
    source = source * scale;
    target = target*scale;
    float4 depth_point_s = { target.x, target.y, target.z, 1.0f };
    auto tX = ppx + fx*(target[0] / target[2]);
    auto tY = ppy + fy*(target[1] / target[2]);

    float4 depth_point_h = { source.x, source.y, source.z, 1.0f };
    auto src_tPt_Frame = linalg::mul(pose, depth_point_h);
    auto sX = ppx + fx*(src_tPt_Frame[0] / src_tPt_Frame[2]);
    auto sY = ppy + fy*(src_tPt_Frame[1] / src_tPt_Frame[2]);

    auto diff = linalg::abs(depth_point_s - src_tPt_Frame);
    auto sqr = [](float x){return x*x; };
    return std::sqrtf(sqr(sX - tX) + sqr(sY - tY));
}

void CalibrationMethod::addImages(std::vector<uint16_t*> depths, std::vector<uint8_t*> colors)
{
    if (n < depths.size()) {
        n = depths.size();
        points.resize(n*n);
        poses.resize(n, linalg::translation_matrix<float>(float3(0, 0, 0)));
    }

    std::vector<std::vector<cv::KeyPoint>> keypoints(n);
    std::vector<cv::Mat> desc(n);
    std::vector<std::vector<float3>> candpoints(n);

    for (int index = 0; index < n; index++){
        //unpack
        auto color = colors[index];
        auto depth = depths[index];
        auto fx = fxs[index];
        auto fy = fys[index];
        auto px = pxs[index];
        auto py = pys[index];

        cv::Mat colorImage(h, w, CV_8UC3, color);
        cv::Mat depthImage(h, w, CV_16U, depth);
        std::vector< cv::KeyPoint > cv_keypoints1;//, cv_keypoints2;
        cv::Mat cv_desc1;
        std::vector<float3> ptCld(h*w);
        //generate point cloud
        for (int y = 0; y < h; y++){
            for (int x = 0; x < w; x++) {
                ptCld[y*w + x] = depth[y*w + x] ? float3((x - px) / fx * depth[y*w + x],
                    (y - py) / fy * depth[y*w + x], depth[y*w + x]) : float3(0.0f, 0.0f, 0.0f);
            }
        }
        auto clampi = [](int x, int b) { return std::max(std::min(x, b - 1), 0); };
        auto cv_detect = cv::ORB();

        cv_detect.detect(colorImage, cv_keypoints1);
        cv_detect.compute(colorImage, cv_keypoints1, cv_desc1);
        keypoints[index] = cv_keypoints1;
        desc[index] = cv_desc1;

        for (int i = 0; i < cv_keypoints1.size(); i++){
            auto kp = cv_keypoints1[i];
            auto yi = clampi((int)std::round(kp.pt.y), h);
            auto xi = clampi((int)std::round(kp.pt.x), w);
            candpoints[index].push_back(ptCld[yi*w + xi]);
        }
    }
    for (int index = 0; index < n; index++){
        for (int index2 = index + 1; index2 < n; index2++) {
            cv::BFMatcher matcher(cv::NORM_HAMMING, true);
            std::vector< cv::DMatch > cv_matches;
            if (desc[index].cols != desc[index2].cols)
                continue;

            matcher.match(desc[index], desc[index2], cv_matches);
            for (const auto & match : cv_matches) {
                auto kypt1 = keypoints[index][match.queryIdx];
                auto kypt2 = keypoints[index2][match.trainIdx];
                auto dist = match.distance;
                auto match1 = candpoints[index][match.queryIdx];
                auto match2 = candpoints[index2][match.trainIdx];
                if (match1.z > 1 && match2.z > 1 && match1.z < Z_THRESH && match2.z < Z_THRESH
                    && dist < ((desc[index].cols * 8 + (SCORE_DIVIDER - 1)) / SCORE_DIVIDER)) {
                    points[n*index + index2].push_back(match1);
                    points[n*index2 + index].push_back(match2);
                }
            }
        }
    }
}

float4x4 CalibrationMethod::computePose(int index1, int index2)
{
    const auto &srcRef = points[n*index1 + index2];
    const auto &dstRef = points[n*index2 + index1];
    std::vector<float> errors(srcRef.size());

    //local copies of points
    //auto srcPoints = points[n*index1 + index2];
    //auto dstPoints = points[n*index2 + index1];

    //storage
    float4x4 bestPose =  linalg::translation_matrix<float>(float3(0, 0, 0));
    float bestPoseErr = INT_MAX;
    int bestPoseIter = -1;

    //defaults
    float4x4 poseLA2 = linalg::translation_matrix<float>(float3(0, 0, 0));

    float reprojError = 0.0f;
    float REPROJ_FILTER = 3e38;
    std::vector<float3> srcPoints, dstPoints;

    int iter_cnt = 0;
    do {
        srcPoints.clear();
        dstPoints.clear();
        //filter points
        for (int i = 0; i < srcRef.size(); i++) {
            auto err = computeReprojectionError(srcRef[i], fxs[index1], fys[index1], pxs[index1], pys[index1], dstRef[i], bestPose);
            if (err < REPROJ_FILTER) {
                srcPoints.push_back(srcRef[i]);
                dstPoints.push_back(dstRef[i]);
            }
        }

        if (srcPoints.size() < 3 || dstPoints.size() != srcPoints.size())
            break;

        Eigen::MatrixXf A(3, srcPoints.size());
        Eigen::MatrixXf B(3, srcPoints.size());
        for (int i = 0; i < srcPoints.size(); i++) {
            A.col(i) = Eigen::Vector3f(srcPoints[i].x, srcPoints[i].y, srcPoints[i].z);
            B.col(i) = Eigen::Vector3f(dstPoints[i].x, dstPoints[i].y, dstPoints[i].z);
        }
        Eigen::Vector3f mean1 = A.rowwise().mean(), mean2 = B.rowwise().mean();
        Eigen::MatrixXf A_zm = A.colwise() - mean1, B_zm = B.colwise() - mean2, C = A_zm*B_zm.transpose();
        Eigen::JacobiSVD< Eigen::MatrixXf> svd = C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3f UVt = svd.matrixU()*svd.matrixV().transpose();
        Eigen::Vector3f v;
        v << 1, 1, UVt.determinant();
        Eigen::Matrix3f R = svd.matrixU()*v.asDiagonal()*svd.matrixV().transpose();
        Eigen::Vector3f t = mean1 - R*mean2;
        Eigen::Affine3f T(R.adjoint());
        T.translation() = -t;
        linalg::aliases::float3x3 laR(R.data());
        linalg::aliases::float3 laT(t.data());
        poseLA2 = linalg::pose_matrix(linalg::rotation_quat(laR), scale*laT);
        //errors.resize(srcPoints.size());
        //for (int i = 0; i < srcPoints.size(); i++) {
        //    errors[i] = computeReprojectionError(srcPoints[i], fxs[index1], fys[index1], pxs[index1], pys[index1], dstPoints[i], poseLA2);
        //}
        //std::sort(errors.begin(), errors.end());
        //reprojError = errors[(int)(errors.size()*0.5f)];

        iter_cnt++;
    
        auto newPoseErr = 0;
        for (int i = 0; i < srcRef.size(); i++) {
            auto err = computeReprojectionError(srcRef[i], fxs[index1], fys[index1], pxs[index1], pys[index1], dstRef[i], poseLA2);
            newPoseErr += err;
        }
        if (newPoseErr < bestPoseErr) {
            bestPoseErr = newPoseErr;
            bestPose = poseLA2;
            bestPoseIter = iter_cnt;
        }
    
        REPROJ_FILTER = std::max(REPROJ_CONV, std::min(bestPoseErr/srcRef.size()-REPROJ_STEP,REPROJ_FILTER));
    } while ((iter_cnt < ITER_MAX && bestPoseErr / srcRef.size() > REPROJ_CONV));
    printf("%d %d %d\t%f\t%d\t%d\t%d\n", bestPoseIter, index1, index2, bestPoseErr / srcRef.size(), iter_cnt, srcPoints.size(), srcRef.size());
    return bestPose;
}

bool CalibrationMethod::solvePose()
{
    using linalg::aliases::int3;
    std::vector<int3> pairs;
    std::vector<int> visited(n, -1);
    std::vector<int> visitedPose(n, -1);

    for (auto & pose : poses) {
        pose = linalg::translation_matrix<float>(float3(0, 0, 0));
    }

    for (int index = 0; index < n; index++){
        for (int index2 = index + 1; index2 < n; index2++) {
            pairs.push_back(int3(index, index2, points[n*index + index2].size()));
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](int3 a, int3 b) { return a.z > b.z;  });
    for (int i = 0; i < pairs.size(); i++) {
        if (pairs[i].z && visited[pairs[i].y] < 0)
            visited[pairs[i].y] = pairs[i].x;
    }
    for (int i = 1; i < visited.size(); i++)
        if (visited[i] < 0)
            return false;

    std::vector<std::vector<int>> pathStacks(visited.size()); // could have one but this is simple and equiv.
    for (int index = 1; index < visited.size(); index++){
        auto target = visited[index];
        pathStacks[index].push_back(target);

        while (target != 0)  {
            target = visited[target];
            pathStacks[index].push_back(target);
        }
    }
    
    for (int index = 0; index < visited.size(); index++) {
        //std::reverse(pathStacks[index].begin(), pathStacks[index].end());
        float4x4 pose = linalg::translation_matrix<float>(float3(0, 0, 0));
        auto prevIdx = index;
        while (prevIdx != 0) {
            pose = linalg::mul(computePose(pathStacks[index].front(), prevIdx),pose);
            prevIdx = pathStacks[index].front();
            pathStacks[index].erase(pathStacks[index].begin());
        }
        poses[index] = pose;
    }

    return true;
}

float4x4 CalibrationMethod::getPose(int index)
{
    return poses[index];
}