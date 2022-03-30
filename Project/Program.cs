using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {

            const string cfgFile = "yolo_v3/yolov3.cfg"; // 정보 파일
            const string darknetModel = "yolo_v3/yolov3.weights"; // model weights
            string[] classNames = File.ReadAllLines("yolo_v3/coco.names"); // detection class matching 

            List<ConsoleColor> color = new List<ConsoleColor>();
            String[] colorNames = ConsoleColor.GetNames(typeof(ConsoleColor));

            List<string> labels = new List<string>();
            List<float> scores = new List<float>();
            List<Rect> bboxes = new List<Rect>();

            // test image
            Mat image = new Mat("./input/bali-crop.jpg");
            Net net = Net.ReadNetFromDarknet(cfgFile, darknetModel);
            Mat inputBlob = CvDnn.BlobFromImage(image, 1 / 255f, new Size(320, 320), crop: false); // 416보다 detection 확률 높음
            //Mat inputBlob = CvDnn.BlobFromImage(image, 1/255f, new Size(416, 416), crop:false);

            net.SetInput(inputBlob);
            var outBlobNames = net.GetUnconnectedOutLayersNames();
            var outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();


            net.Forward(outputBlobs, outBlobNames);
            foreach (Mat prob in outputBlobs)
            {
                for (int p = 0; p < prob.Rows; p++)
                {
                    float confidence = prob.At<float>(p, 4);
                    if (confidence > 0.6)
                    {
                        Cv2.MinMaxLoc(prob.Row(p).ColRange(5, prob.Cols), out _, out _, out _, out Point classNumber);

                        int classes = classNumber.X;
                        float probability = prob.At<float>(p, classes + 5);

                        // 임계값 설정
                        if (probability > 0.5)
                        {
                            // Bounding Box Center Pose & Szie Buffer
                            // 바운딩 박스 중심 좌표 & 박스 크기
                            float centerX = prob.At<float>(p, 0) * image.Width;
                            float centerY = prob.At<float>(p, 1) * image.Height;
                            float width = prob.At<float>(p, 2) * image.Width;
                            float height = prob.At<float>(p, 3) * image.Height;

                            // detection 위치
                            int sx = ((int)centerX - (int)width / 2);
                            int sy = ((int)centerY - (int)height / 2);

                            labels.Add(classNames[classes]);
                            scores.Add(probability);
                            
                            bboxes.Add(new Rect(sx, sy, (int)width, (int)height));
                            //bboxes.Add(new Rect((int)centerX - (int)width / 2, (int)centerY - (int)height / 2, (int)width, (int)height));
                        }
                    }
                }
            }
            // bounding box 정보, 검출율, 임계값 ~, detection 정보
            CvDnn.NMSBoxes(bboxes, scores, 0.5f, 0.4f, out int[] indices);
            
            foreach (int i in indices)
            {
                Cv2.Rectangle(image, bboxes[i], Scalar.RandomColor(), 1);
                //Cv2.Rectangle(image, bboxes[i], Scalar.Red, 1);
                Cv2.PutText(image, labels[i], bboxes[i].Location, HersheyFonts.HersheyComplex, 1.0, Scalar.Red);
            }

            Cv2.ImShow("image", image);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();
        }
    }
}