%% Variable declaration
person = 40;
sampleFromEachPersonTest = 10 - sampleFromEachPerson;
imgRow = 112;
imgCol = 92;
testDb = zeros(person*sampleFromEachPersonTest, imgRow*imgCol);

%% Read data
count = 1;
for i=1:person
    for j=(sampleFromEachPerson+1):(sampleFromEachPerson+sampleFromEachPersonTest)
        fname = strcat('s', int2str(i), '/', int2str(j),'.pgm');
        img = imread(fname);
        testDb(count, :) = (img(:))';
        count = count + 1;
    end
end

%% Perform PCA
testDb = bsxfun(@minus,testDb,mean(testDb, 2));
[eigenvectorsTest, score,latent,tsquared,explained] = pca(testDb);

%% Show PCA result

for i=1:5
    cur = (eigenvectorsTest(:,i)) * 256;
    img = reshape(cur, [imgRow, imgCol]);
    imshow(img);
    %pause;
end

%% Transform dataset

transformedDbTest = testDb * eigenvectorsTest(:, 1:m);

%% Form desicion tree

testLabel = zeros(person*sampleFromEachPersonTest, 1);
count = 0;
for i=1:person
    start = (count*sampleFromEachPersonTest)+1;
    finish = (count+1)*sampleFromEachPersonTest;
    testLabel(start:finish) = i;
    count = count + 1;
end

%% Predict

classificationLabel = predict(tree, transformedDbTest);

%% Accuracy

result = (testLabel == classificationLabel);
accuracy = (sum(result == 1))/(person*sampleFromEachPersonTest);
accuracy = accuracy * 100
