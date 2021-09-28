function stack=imreadTiff(filename)
info = imfinfo(filename);
frames = numel(info);
stack=zeros(info(1).Height,info(1).Width,frames, 'uint8');
for k = 1:frames
    stack(:,:,k) =imread(filename, k);
end