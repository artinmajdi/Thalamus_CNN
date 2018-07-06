clear
clc
close all

% Load the trained model 
net = vgg16();

% See details of the architecture 
net.Layers

% Read the image to classify 
I = imread('peppers.png');

% Adjust size of the image 
sz = net.Layers(1).InputSize;
I = I(1:sz(1),1:sz(2),1:sz(3));

% Classify the image using VGG-16 
label = classify(net, I);

% Show the image and the classification results 
figure; 
imshow(I) 
text(10, 20, char(label),'Color','white')