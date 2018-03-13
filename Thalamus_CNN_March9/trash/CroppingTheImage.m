clear
clc
close all

addpath('../NIfTI_20140122/')

DirectoryData = '../../data/new20Images/';
mskFull = load_nii([DirectoryData,'mask_templ_93x187x68.nii.gz']);
msk = mskFull.img>0;

Dir = dir(DirectoryData);
Dir = Dir(3:end);

for i = 1%:length(Dir)
   Directory = [Dir(i).folder,'\',Dir(i).name,'\'];
   ThalamusSeg = load_nii([Directory,'WholeThalamusSegment_TemplateDomain.nii.gz']);
   ThalamusSeg = ThalamusSeg.img;
   ImgDeformed = load_nii([Directory,'WMnMPRAGEdeformed.nii.gz']);
   ImgDeformed = ImgDeformed.img;
   
   ThalamusSegCroped = single(msk*0);
   ImgDeformedCroped = single(msk*0);
   ThalamusSegCroped(msk) = ThalamusSeg(msk);
   ImgDeformedCroped(msk) = ImgDeformed(msk);
   
   save_nii(ThalamusSegCroped, [Directory,'WholeThalamusSegment_TemplateDomain_Croped.nii'])
   save_nii(ImgDeformedCroped, [Directory,'WMnMPRAGEdeformed_Croped.nii'])
   
end