%% Variable declaration
person = 40;
imgRow = 112;
imgCol = 92;
trainDb = zeros(person*sampleFromEachPerson, imgRow*imgCol);

%% Read data
count = 1;
for i=1:person
    for j=1:sampleFromEachPerson
        fname = strcat('s', int2str(i), '/', int2str(j),'.pgm');
        img = imread(fname);
        trainDb(count, :) = (img(:))';
        count = count + 1;
    end
end

%% Perform PCA
trainDb = bsxfun(@minus,trainDb,mean(trainDb, 2));
[eigenvectors, score,latent,tsquared,explained] = pca(trainDb);

%% Show PCA result

for i=1:5
    cur = (eigenvectors(:,i)) * 256;
    img = reshape(cur, [imgRow, imgCol]);
    imshow(img);
    %pause;
end

%% Transform dataset

transformedDb = trainDb * eigenvectors(:, 1:m);

%% Form desicion tree

trainLabel = zeros(person*sampleFromEachPerson, 1);
count = 0;
for i=1:person
    start = (count*sampleFromEachPerson)+1;
    finish = (count+1)*sampleFromEachPerson;
    trainLabel(start:finish) = i;
    count = count + 1;
end

tree = fitctree(transformedDb, trainLabel);
