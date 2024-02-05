// This is how I call the function (z is the depth image):
// params = [];
// params = estimateBackscatter(RGB,z,params,10);
// These are the functions:
function [params,thisz,minvals] = estimateBackscatter(I,z,params,pdark)
if nargin<4
% Most reliable dark points
pdark = 1;
end
% Subsample both
Ism = imresize(I,[300 NaN]);
Ism(Ism<0) = 0;
zsm = imresize(z,[300 NaN]);
zsm(zsm<0) = NaN;
[darkZ,Bc,~,~] = getBackscatterByCurveFittingMultipleImages(Ism,zsm,pdark);
% Divide into 20 bins
ints = linspace(min(darkZ(:)),max(darkZ(:)),20);
minvals = nan(numel(ints)-1,3);
for k = 1:3
for i = 1:numel(ints)-1
% What's the minimum of this interval?
ind = (darkZ > ints(i)) & (darkZ <= ints(i+1));
this_val = min(Bc(ind,k));
%this_val = max(Bc(ind,k));
if ~isempty(this_val)
minvals(i,k) = this_val;
end
end
end
thisz = ints(1:end-1);
exclude = isnan(minvals(:,1)) ;
thisz(exclude) = [];
minvals(exclude,:) = [];
thisz = [0;thisz'];
minvals = [0 0 0; minvals];
% % optimization
x0 = randn(2,1);
lb = [0 0];
ub = [1 1];
options = optimoptions(@fmincon,'Display', 'none', 'MaxIter', 1000,'MaxFunEvals',100000,...
'TolFun', 1e-6, 'DiffMinChange', 1e-6, 'TolX', 1e-6,'useParallel',false);
for k = 1:3
thisVal = thisz;
f = @(x)parameter_fun_Bc_diff2(x,thisz,minvals(:,k));
g = @(x)constraint_fun_Bc_diff(x,thisz,minvals(:,k));
problem = createOptimProblem('fmincon','objective',f,'x0',x0,'lb',lb,'ub',ub,'options',options,'nonlcon',g);
out(:,k) = fmincon(problem);
end
Bc_inf = out(1,:);
bcb = out(2,:);
params.Bc_inf = Bc_inf;
params.betac_b = bcb;
params.pdark = pdark;

function [darkZ,Bc,pdark,maskRho] = getBackscatterByCurveFittingMultipleImages(I,dm,pdark,rhoflag)
if nargin<3
rhoflag = 0;
pdark = 1;
end
if nargin<4
rhoflag = 0;
end
% getBackscatterByCurveFittingMultipleImages This function computes an
% initial (over) estimate of Backscatter from a give image.
s = size(I);
if rhoflag
% This is the width of the Gaussian
sigmaVal = 1*(max(s(1),s(2)))./2;
% % Illumination map
Ec = 2*imgaussfilt(I,sigmaVal,'filterdomain','spatial');
% A rough estimate of reflectance, or a type of intrinsic image.
rho = I./Ec;
rho = mat2gray(rho);
else
rho = I;
end
% Find the darkest pixel triplets of rho in the "pdark" percentile.
% Get the values of the darkest pixels in Ic, but check each range bracket
edges = linspace(min(dm(:)),max(dm(:)),10);
zcluster = clusterRange(dm,edges);
maskRho = zeros(s(1),s(2));
for i = 1:numel(edges)-1
thisMask = (zcluster >=edges(i)) & (zcluster < edges(i+1));
if sum(thisMask(:)) >0
thisRho = cropImgToDepthMap(rho,~thisMask);
thisMaskRho = findDarkestPixels(thisRho,pdark);
numFound = sum(thisMaskRho(:));
% while sum(thisMaskRho(:))< 0.01*numFound
% pdark = pdark + 0.5;
% thisMaskRho = findDarkestPixels(thisRho,pdark);
% end
maskRho = maskRho + thisMaskRho;
end
end
maskRho = logical(maskRho);
[darkZ,Bc] = extractDarkestPixels(I,maskRho,dm);
% Exclude nans or infs
exclude2 = isnan(darkZ) | isinf(darkZ);
darkZ(exclude2) = [];
Bc(exclude2,:) =[];
exclude = Bc(:,1) < 0 | Bc(:,1) < 0 | Bc(:,1) < 0;
darkZ(exclude) = [];
Bc(exclude,:) =[];

function y = parameter_fun_Bc_diff2(x,z,Ic)
Bc_inf = x(1);
bcb = x(2);
y = norm(Ic - (Bc_inf.*(1-exp(-bcb.*z)))');


function [darkZ,darkPix] = extractDarkestPixels(I,mask,dm)
s = size(I);
if numel(s) ==3
Ires = reshape(I,[s(1)*s(2) 3]);
maskres = reshape(mask,[s(1)*s(2) 1]);
dmres = reshape(dm,[s(1)*s(2) 1]);
darkPix = Ires(maskres,:);
darkZ = dmres(maskres);
else
Ires = reshape(I,[s(1)*s(2) 1]);
maskres = reshape(mask,[s(1)*s(2) 1]);
dmres = reshape(dm,[s(1)*s(2) 1]);
darkPix = Ires(maskres,:);
darkZ = dmres(maskres);
end