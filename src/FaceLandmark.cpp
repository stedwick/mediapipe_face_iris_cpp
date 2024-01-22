#include "FaceLandmark.hpp"
#include <iostream>

#include <ApplicationServices/ApplicationServices.h>

#define FACE_LANDMARKS 468
/*
Helper function
*/
bool __isIndexValid(int idx)
{
    if (idx < 0 || idx >= FACE_LANDMARKS)
    {
        std::cerr << "Index " << idx << " is out of range ("
                  << FACE_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}

my::FaceLandmark::FaceLandmark(std::string modelPath) : FaceDetection(modelPath),
                                                        m_landmarkModel(modelPath + std::string("/face_landmark.tflite"))
{
}

void my::FaceLandmark::runInference()
{
    if (!gotRoi)
    {
        FaceDetection::runInference();
        roi = FaceDetection::getFaceRoi();
        if (roi.empty())
            return;
        gotRoi = true;
    }

    auto face = FaceDetection::cropFrame(roi);
    m_landmarkModel.loadImageToInput(face);
    m_landmarkModel.runInference();
}

cv::Point my::FaceLandmark::getFaceLandmarkAt(int index) const
{
    if (__isIndexValid(index))
    {
        auto roi = FaceDetection::getFaceRoi();

        float _x = m_landmarkModel.getOutputData()[index * 3];
        float _y = m_landmarkModel.getOutputData()[index * 3 + 1];

        int x = (int)(_x / m_landmarkModel.getInputShape()[2] * roi.width) + roi.x;
        int y = (int)(_y / m_landmarkModel.getInputShape()[1] * roi.height) + roi.y;

        return cv::Point(x, y);
    }
    return cv::Point();
}

std::vector<cv::Point> my::FaceLandmark::getAllFaceLandmarks() const
{
    if (FaceDetection::getFaceRoi().empty())
        return std::vector<cv::Point>();

    std::vector<cv::Point> landmarks(FACE_LANDMARKS);
    for (int i = 0; i < FACE_LANDMARKS; ++i)
    {
        auto mark = getFaceLandmarkAt(i);
        landmarks[i] = mark;
        if (i == 8)
        {
            if (mouse_x != 0)
            {
                auto diff_x = mark.x - mouse_x;
                auto diff_y = mark.y - mouse_y;

                if (std::pow(diff_y,2) > 0 && std::pow(diff_x,2) > 0)
                {
                    CGEventRef eventGet = CGEventCreate(NULL);
                    CGPoint cursor = CGEventGetLocation(eventGet);
                    CFRelease(eventGet);
                    // std::cout << cursor.x << ", " << cursor.y << "\n\n";

                    CGPoint location = CGPointMake(cursor.x + diff_x*3, cursor.y + diff_y*4);
                    CGEventRef eventSet = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, location, kCGMouseButtonLeft);
                    CGEventSetType(eventSet, kCGEventMouseMoved);
                    CGEventPost(kCGHIDEventTap, eventSet);
                    CFRelease(eventSet);
                }
            }
            mouse_x = mark.x;
            mouse_y = mark.y;
        }
    }
    return landmarks;
}

std::vector<float> my::FaceLandmark::loadOutput(int index) const
{
    return m_landmarkModel.loadOutput();
}