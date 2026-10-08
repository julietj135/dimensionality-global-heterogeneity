%% THIS SCRIPT CREATES FIGURES IN Operator-valued free probability applied to recurrent neural networks with heterogeneity in local and global circuit structures
% Juliet Jiang

set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');   % covers title, xlabel, ylabel, text()
set(groot, 'defaultLegendFontName', 'Arial'); % if you use legends
set(groot, 'defaultColorbarFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 14);
set(groot, 'defaultAxesLineWidth', 2);
set(groot, 'defaultLineLineWidth', 2);

%% Figure 5a. Distributions of eigenvalues of C and R

rng(42);
N = 1000;
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
W = randn(N)*g0 / sqrt(N);

% change DR here
DR = 0.55;
[R, mu, sigma] = simr(targetMean, DR, N); % Generate firing rates
temp = diag(R.^(-1/2)) - W;
E1 = svd(temp).^(-2);
disp(pr(E1))

% obtain density of eigenvalues and density of rates
x = logspace(0, 1.7, 1200);
p_old = computepdf(x, R, g0);
p_R = lognpdf(x, mu, sigma);

figure;
histogram(E1, "Normalization","pdf",'EdgeColor', 'none', 'NumBins', 500); hold on;
plot(x,p_old,'k','LineWidth',1);
xlabel('Covariance eigenvalues');
xscale("log")
yticks([])
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,300,200])
box off;

figure;
histogram(R, "Normalization","pdf",'EdgeColor', 'none', 'NumBins', 100); hold on;
plot(x,p_R,'k','LineWidth',1);
xscale("log")
xlabel('Firing rates');
yticks([])
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,300,200])
box off;

%% Figure 5b. Linear relationship in unstructured network

rng(42)
DRs = 0:0.1:1;
g = 0.6*gm;
Rmeans = 19:-2:5;
cmap = sky(length(Rmeans));

nR = 100; % number of operating points
targetMean = 5; 
DC = zeros(1, nR); DR = zeros(1, nR); 
h = waitbar(0);

% get W without any added structure
W = get_W_oneblock(1000, 0, 4*g^2, true, "bottom right"); 
for iR = 1 : nR
    targetPR = rand * 0.8 + 0.1; % sample D(R) uniformly between 0.1 and 0.9
    [R, E] = simrc(W, targetMean, targetPR);
    DR(iR) = pr(R); DC(iR) = pr(E);
    waitbar(iR/nR, h)
end
delete(h)
figure; 
scatter(DR,DC,25,'k','filled'); hold on;
for i=1:length(Rmeans)
    meanR = Rmeans(i);
    DCs = DRs*(1-g^2*meanR)^2;
    currentColor = cmap(i, :); 
    plot(DRs, DCs,'Color', currentColor, 'LineWidth', 2); 
end
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,300,400])
xlabel('$\textsf{Firing rate uniformity }D(R)$', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$\textsf{Ef{}fective Dimension }D(C)$', 'FontSize', 15, 'Interpreter', 'latex');
box off;
% saveas(f,"figures_parts/fig2/DC_DR_sweep.pdf")

%% Figure 5c. Moments of the covariance in an unstructured network with a parameter sweep

% parameter sweep
g0s = 0.2:0.001:0.5; 
meanRs = 5:0.01:15;
DR = 0.8;
secondRs = meanRs.^2/DR;

nG = length(g0s);
nM = length(meanRs);
phiC  = zeros(nG, nM);
phiC2 = zeros(nG, nM);  
DCs = zeros(nG, nM); 
stable = false(nG, nM);  % stability mask: g0*sqrt(meanR) < 1

for iG = 1:nG
    g0 = g0s(iG);
    for iM = 1:nM
        meanR = meanRs(iM);
        phiR  = meanR;
        phiR2 = secondRs(iM); 

        % stability condition
        stable(iG, iM) = (g0 * sqrt(meanR)) < 0.99;

        denom = 1 - g0^2 * phiR; % shared denominator term

        phiC(iG, iM)  = phiR  / denom;
        phiC2(iG, iM) = phiR2 / denom^4;
        DCs(iG, iM) = phiC(iG, iM)^2/phiC2(iG, iM);

    end
end

% masked versions (NaN outside the stable/linearizable regime)
phiC_masked  = phiC;
phiC2_masked = phiC2;
DCs_masked = DCs;
phiC_masked(~stable)  = NaN;
phiC2_masked(~stable) = NaN;
DCs_masked(~stable) = NaN;

% stability boundary curve: g0 = 1/sqrt(meanR), plotted over meanRs range
g0_boundary = 1 ./ sqrt(meanRs);

darkToLightBlue = @(n) [linspace(0.02,0.75,n)', linspace(0.02,0.85,n)', linspace(0.35,1,n)'];
figure('position', [100, 100, 600, 400]);
subplot(1,3,1);
imagesc(meanRs, g0s, phiC_masked);
set(gca, 'YDir', 'normal');
set(gca, 'ColorScale', 'log')
set(gca, 'FontSize', 14);
hold on;
plot(meanRs, g0_boundary, 'r-', 'LineWidth', 2);   % stability boundary: g0*sqrt(meanR)=1
xlabel('$\varphi[R]$','Interpreter','latex','FontSize', 18);
ylabel('$g_0$','Interpreter','latex','FontSize', 20);
t = title("First moment",'FontSize', 14);
t.Position(2) = t.Position(2) + 0.01;
colormap(gca, darkToLightBlue(nG));
cb = colorbar('southoutside'); cb.LineWidth = 2;
xlim([min(meanRs) max(meanRs)]); ylim([min(g0s) max(g0s)]);
set(gca, 'LineWidth', 2);

subplot(1,3,2);
imagesc(meanRs, g0s, phiC2_masked);
set(gca, 'YDir', 'normal');
set(gca, 'ColorScale', 'log')
set(gca, 'FontSize', 14);
hold on;
plot(meanRs, g0_boundary, 'r-', 'LineWidth', 2);
xlabel('$\varphi[R]$','Interpreter','latex','FontSize', 18);
t = title("Second moment",'FontSize', 14);
t.Position(2) = t.Position(2) + 0.01;
colormap(gca, darkToLightBlue(nG));
cb = colorbar('southoutside'); cb.LineWidth = 2;
xlim([min(meanRs) max(meanRs)]); ylim([min(g0s) max(g0s)]);
set(gca, 'LineWidth', 2);

subplot(1,3,3);
imagesc(meanRs, g0s, DCs_masked);
set(gca, 'YDir', 'normal');
set(gca, 'ColorScale', 'log')
set(gca, 'FontSize', 14);
hold on;
plot(meanRs, g0_boundary, 'r-', 'LineWidth', 2);
xlabel('$\varphi[R]$','Interpreter','latex','FontSize', 18);
t = title("Dimension",'FontSize', 14);
t.Position(2) = t.Position(2) + 0.01;
colormap(gca, darkToLightBlue(nG));
cb = colorbar('southoutside'); cb.LineWidth = 2;
xlim([min(meanRs) max(meanRs)]); ylim([min(g0s) max(g0s)]);
set(gca, 'LineWidth', 2);

%% Figure 6b. Eigenvalue distributions of unstructured vs within-area structure

% set hyperparameters and get total variance and R
rng(45)
N0 = 2000; 
N = N0*2;
targetMean = 10; 
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
DR = 0.8; % D(R)

% amount of added structure
eps = sqrt(0.07);
g = sqrt(g2total-eps^2)/2;

x = linspace(0.1, 600, 2000);
R = simr(targetMean, DR, N); 

trials = 10;
Es = zeros(N);
for j = 1:trials
    W = get_W_oneblock(N0, eps, g2total, conserved, "bottom right");
    % bet simulated distribution of evals of C
    temp = diag(R.^(-1/2)) - W;
    E = svd(temp).^(-2);
    Es = Es + sort(E, 'descend');
end
E = Es/trials; 
p = computepdf_eps(x, R, g, eps); 
p_old = computepdf(x, R, g0);

% get eigenvalues by rank
c = cumsum(p); c = c / c(end);
c_old = cumsum(p_old); c_old = c_old / c_old(end);
nRank = 40; 
Edata = sort(E, 'descend'); Edata = Edata(1 : nRank);
ratio = 10; nRankT = nRank * ratio; 
Etheory_old = zeros(1, nRankT); Etheory = zeros(1, nRankT);
for iRank = 1 : nRankT
    Etheory_old(iRank) = x(find(c_old>1-iRank/ratio/N, 1));
    Etheory(iRank) = x(find(c>1-iRank/ratio/N, 1));
end

% plot largest eigenvalues by rank
figure;
plot(1/ratio:1/ratio:nRank, Etheory_old,'k','LineWidth',2); hold on
scatter(1:nRank, Edata,36,[0 0.4470 0.7410], 'filled');
plot(1/ratio:1/ratio:nRank, Etheory,'r','LineWidth',2);  
% legend("Homogeneous","Perturbed","Simulation")
xlabel('Rank'); ylabel('Eigenvalue')
set(gca,'xtick',[])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,400,250])
box off;

% plot eigenvalue distribution
figure;
h=histogram(E,2000,'Normalization','pdf'); hold on;
plot(x,p_old,'k','LineWidth',2);
plot(x,p,'r','LineWidth',2);
h.EdgeColor = 'none';
set(gca, 'XScale', 'log')
legend("Sim","Homogeneous","Within-area","Location","northoutside",'Orientation', 'horizontal')
xlabel("Eigenvalue")
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
legend boxoff;
set(gcf,'position',[100,100,400,400])

%% Figure 6c and 6d (left panel). Statistics of the covariance eigenvalue spectrum with within-area structure
% Also Figure S2a. Statistics with within-area structure without conserved variance

rng(42)

% parameters
N0 = 500;
N = 2*N0;
DR = 0.8; 
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
R = simr(targetMean, DR, N); % generate firing rates
a2 = mean(R);
a4 = mean(R.^2);
DR = pr(R);

% change for case with unconserved variance
conserved = true;

eps_2_list = 0:0.002:0.1;
ratios = zeros(length(eps_2_list),1);
means_sim = zeros(length(eps_2_list),1); 
means_theory = zeros(length(eps_2_list),1); 
means_theory_ex2 = zeros(length(eps_2_list),1);
var_sim = zeros(length(eps_2_list),1); 
var_theory = zeros(length(eps_2_list),1);
var_theory_ex2 = zeros(length(eps_2_list),1);
DC_sim = zeros(length(eps_2_list),1);
DC_theory = zeros(length(eps_2_list),1);
DC_theory_ex2 = zeros(length(eps_2_list),1);
for i = 1:length(eps_2_list)
    eps = sqrt(eps_2_list(i));
    disp(eps_2_list(i))

    if conserved
        g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    else
        g = sqrt(g2total)/2; % variance not conserved
    end

    ratios(i) = (g^2+eps^2)/g^2;

    means = 0;
    vars = 0;
    dcs = 0;
    trials = 1;
    for j = 1:trials
        % get moments from simulated network
        W = get_W_oneblock(N0, eps, g2total, conserved, "bottom right");
        temp = diag(R.^(-1/2)) - W;
        E = svd(temp).^(-2);
        means = means + mean(E);
        vars = vars + mean(E.^2);
        dcs = dcs + pr(E);
    end
    % simulated statistics, trial averaged
    means_sim(i) = means/trials;
    var_sim(i) = vars/trials;
    DC_sim(i) = dcs/trials;

    % structured theory
    means_theory(i) = (a2*(4 - a2* eps^2))/(4 + a2^2* eps^2 *g^2 - 2 *a2* (eps^2 + 2 *g^2)); % variance conserved, exact
    var_theory(i) = (8*a4*(16 + (2 - eps^2 *a2)^4))/(4 + a2^2 *eps^2* g^2 - 2* a2 *(eps^2 + 2* g^2))^4;
    DC_theory(i) = (a2^2*(-4 + a2 *eps^2)^2*(4 + a2^2 *eps^2* g^2 - 2* a2* (eps^2 + 2 *g^2))^2)/(8* a4* (16 + (-2 + a2* eps^2)^4));
    
    % unstructured theory
    means_theory_ex2(i) = a2/(1 - a2*(g^2+eps^2/4));
    var_theory_ex2(i) = a4*(1/(1-a2*(g^2+eps^2/4))^4);
    DC_theory_ex2(i) = DR * (1 - (g^2+eps^2/4) * a2)^2; 
end

% figures
figure;
scatter(ratios, means_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, means_theory, "r",LineWidth=2);
plot(ratios, means_theory_ex2, "k",LineWidth=2);
% legend("Simulation","Within-area","Homogoneous","Location", "northoutside",'Orientation', 'horizontal');
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('First moment','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
% legend boxoff;
set(gcf,'position',[100,100,300,400])

figure;
scatter(ratios, var_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, var_theory, "r",LineWidth=2);
plot(ratios, var_theory_ex2, "k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('Second moment','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

figure;
scatter(ratios, DC_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, DC_theory, "r",LineWidth=2);
plot(ratios, DC_theory_ex2, "k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('D(C)','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

%% Figure 6d (right panel). D(C) and D(R) relationship with within-area structure

rng(42)

% parameters
N0 = 1000;
N = 2*N0;
nR = 100; % Number of conditions (i.e. operating points)
targetMean = 10; 
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;
DC = zeros(1, nR); DR = zeros(1, nR); % Initialize D(C) and D(R)

conserved = true;

% added structure
eps = sqrt(0.07);
g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
disp(eps^2)

h = waitbar(0);
W = get_W_oneblock(N0, eps, g2total, conserved, "bottom right");
for iR = 1 : nR
    targetPR = rand * 0.8 + 0.1; % Sample D(R) uniformly between 0.1 and 0.9
    [R, E] = simrc(W, targetMean, targetPR);
    DR(iR) = pr(R); DC(iR) = pr(E);
    waitbar(iR/nR, h)
end
delete(h)
x = 0 : 0.1 : 1; 
y = x * (1 - (g^2+eps^2/4) * targetMean)^2; % theoretical prediction
a2 = targetMean;
a4 = targetMean^2./x;

y_new = (a2.^2*(-4 + a2 *eps^2).^2*(4 + a2.^2 *eps^2* g^2 - 2* a2* (eps^2 + 2 *g^2))^2)./(8* a4* (16 + (-2 + a2* eps^2).^4));

% plot result
figure;
scatter(DR, DC, 25, [0 0.4470 0.7410], "filled"); hold on
plot(x, y,'k','LineWidth',2);
plot(x, y_new,'r','LineWidth',2); 
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])
load('axes_position.mat', 'pos')
ax = gca;
set(ax, 'Position', pos)
xlabel('$\textsf{Firing rate uniformity }D(R)$', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$\textsf{Ef{}fective Dimension }D(C)$', 'FontSize', 15, 'Interpreter', 'latex');

%% Figure 7b. Eigenvalue distributions of unstructured vs between-area structure

rng(42)
N0 = 2000; 
N = N0*2;
targetMean = 10; 
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
DR = 0.8; 

eps = sqrt(0.11);
g = sqrt(g2total-eps^2)/2;

x = linspace(0.1, 300, 4000);
R = simr(targetMean, DR, N); 

conserved = true;

trials = 10;
Es = zeros(N);
for j = 1:trials
    disp(j)
    W = get_W_oneblock(N0, eps, g2total, conserved, "top right");
    temp = diag(R.^(-1/2)) - W;
    E = svd(temp).^(-2);
    Es = Es + sort(E, 'descend');
end
E = Es/trials; 
p = computepdf_eps2(x, R, g, eps); 
p_old = computepdf(x, R, g0);

% get eigenvalues by rank 
c = cumsum(p); c = c / c(end);
c_old = cumsum(p_old); c_old = c_old / c_old(end);
nRank = 0; 
Edata = sort(E, 'descend'); Edata = Edata(1 : nRank);
ratio = 10; nRankT = nRank * ratio; 
Etheory_old = zeros(1, nRankT); Etheory = zeros(1, nRankT);
for iRank = 1 : nRankT
    Etheory_old(iRank) = x(find(c_old>1-iRank/ratio/N, 1));
    Etheory(iRank) = x(find(c>1-iRank/ratio/N, 1));
end

% plot largest eigenvalues by rank
figure;
plot(1/ratio:1/ratio:nRank, Etheory_old,'k','LineWidth',2); hold on
scatter(1:nRank, Edata,36,[0 0.4470 0.7410], 'filled');
plot(1/ratio:1/ratio:nRank, Etheory,'r','LineWidth',2);  
xlabel('Rank'); ylabel('Eigenvalue')
set(gca,'xtick',[])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2)
set(gcf,'position',[100,100,300,200])
box off;

% plot eigenvalue distribution
figure;
h=histogram(E,2000,'Normalization','pdf'); hold on;
plot(x,p_old,'k','LineWidth',2);
plot(x,p,'r','LineWidth',2);
h.EdgeColor = 'none';
set(gca, 'XScale', 'log')
legend("Sim","Homogeneous","Btwn-area","Location","northoutside",'Orientation', 'horizontal')
xlabel("Eigenvalue")
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
legend boxoff;
set(gcf,'position',[100,100,400,400])

%% Figure 7c and 7d (left panel). Statistics of the covariance eigenvalue spectrum with between-area structure
% Also Figure S2b. Statistics with between-area structure without conserved variance

rng(42)
N0 = 1000;
N = 2*N0;
DR = 0.8; % D(R)
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
R = simr(targetMean, DR, N); % generate firing rates
a2 = mean(R);
a4 = mean(R.^2);
DR = pr(R);

% change for case with unconserved variance
conserved = false;

eps_2_list = 0:0.002:0.1;
ratios = zeros(length(eps_2_list),1);
means_sim = zeros(length(eps_2_list),1); 
means_theory = zeros(length(eps_2_list),1); 
means_theory_ex2 = zeros(length(eps_2_list),1);
var_sim = zeros(length(eps_2_list),1); 
var_theory = zeros(length(eps_2_list),1);
var_theory_ex2 = zeros(length(eps_2_list),1);
DC_sim = zeros(length(eps_2_list),1);
DC_theory = zeros(length(eps_2_list),1);
DC_theory_ex2 = zeros(length(eps_2_list),1);
for i = 1:length(eps_2_list)
    eps = sqrt(eps_2_list(i));
    disp(eps_2_list(i))

    if conserved
        g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    else
        g = sqrt(g2total)/2; % variance not conserved
    end

    ratios(i) = (g^2+eps^2)/g^2;

    means = 0;
    vars = 0;
    dcs = 0;
    trials = 10;
    for j = 1:trials
        % get moments from simulated network
        W = get_W_oneblock(N0, eps, g2total, conserved, "top right");
        temp = diag(R.^(-1/2)) - W;
        E = svd(temp).^(-2);
        means = means + mean(E);
        vars = vars + mean(E.^2);
        dcs = dcs + pr(E);
    end
    % simulated statistics, trial averaged
    means_sim(i) = means/trials;
    var_sim(i) = vars/trials;
    DC_sim(i) = dcs/trials;

    % structured theory
    means_theory(i) = (a2 * (1 + (eps^2 / 4) * a2)) / (1 - g^2 * a2 * (1 + (eps^2 / 4) * a2));
    var_theory(i) = (a4 * (1 + (eps^2 / 2) * a2)^2) / ((1 - g^2 * a2 * (1 + (eps^2 / 4) * a2))^4); 
    DC_theory(i) = (a2^2 * (1 + (eps^2 / 4) * a2)^2 * (1 - g^2 * a2 * (1 + (eps^2 / 4) * a2))^2) / (a4 * (1 + (eps^2 / 2) * a2)^2);

    % unstructured theory
    means_theory_ex2(i) = a2/(1 - a2*(g^2+eps^2/4));
    var_theory_ex2(i) = a4*(1/(1-a2*(g^2+eps^2/4))^4);
    DC_theory_ex2(i) = DR * (1 - (g^2+eps^2/4) * a2)^2; 
end

% figures
figure;
scatter(ratios, means_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, means_theory, "r",LineWidth=2);
plot(ratios, means_theory_ex2, "k",LineWidth=2);
% legend("Simulation","Within-area","Homogoneous","Location", "northoutside",'Orientation', 'horizontal');
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('First moment','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
% legend boxoff;
set(gcf,'position',[100,100,300,400])

figure;
scatter(ratios, var_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, var_theory, "r",LineWidth=2);
plot(ratios, var_theory_ex2, "k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('Second moment','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

figure;
scatter(ratios, DC_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, DC_theory, "r",LineWidth=2);
plot(ratios, DC_theory_ex2, "k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('D(C)','fontsize',16);
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

%% Figure 7d (right panel). D(C) and D(R) relationship with between-area structure

rng(42)

% parameters
N0 = 1000;
nR = 100; % number of operating points
targetMean = 10; 
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;
DC = zeros(1, nR); DR = zeros(1, nR); % Initialize D(C) and D(R)

conserved = true;

eps = sqrt(0.11);
disp(eps^2)
g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    
h = waitbar(0);
W = get_W_oneblock(N0, eps, g2total, conserved, "top right");
for iR = 1 : nR
    targetPR = rand * 0.8 + 0.1; % Sample D(R) uniformly between 0.1 and 0.9
    [R, E] = simrc(W, targetMean, targetPR);
    DR(iR) = pr(R); DC(iR) = pr(E);
    waitbar(iR/nR, h)
end
delete(h)
x = 0.001 : 0.1 : 1; 
y = x * (1 - (g^2+eps^2/4) * targetMean)^2; % theoretical prediction
a2 = targetMean;
a4 = targetMean^2./x;
y_new = (a2.^2 * (1 + (eps^2 / 4) * a2).^2 * (1 - g^2 * a2 * (1 + (eps^2 / 4) * a2)).^2) ./ (a4 * (1 + (eps^2 / 2) * a2).^2);

figure;
scatter(DR, DC, 25, [0 0.4470 0.7410], "filled"); hold on
plot(x, y,'k','LineWidth',2);
plot(x, y_new,'r','LineWidth',2); 
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])
load('axes_position.mat', 'pos')
ax = gca;
set(ax, 'Position', pos)
xlabel('$\textsf{Firing rate uniformity }D(R)$', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$\textsf{Ef{}fective Dimension }D(C)$', 'FontSize', 15, 'Interpreter', 'latex');

%% Figure 8b (first and second panels). Statistics of the covariance eigenvalue spectrum with both within-area and between-area structure

rng(42)

% parameters
N0 = 2000;
N = 2*N0;
DR = 0.8; % D(R)
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
R = simr(targetMean, DR, N); % generate firing rates
a2 = mean(R);
a4 = mean(R.^2);
DR = pr(R);
conserved = true;

eps_2_list = 0:0.002:0.1;
ratios = zeros(length(eps_2_list),1);
means_sim = zeros(length(eps_2_list),1); 
means_theory = zeros(length(eps_2_list),1); 
means_theory_ex2 = zeros(length(eps_2_list),1);
var_sim = zeros(length(eps_2_list),1); 
var_theory = zeros(length(eps_2_list),1);
var_theory_ex2 = zeros(length(eps_2_list),1);
for i = 1:length(eps_2_list)
    eps = sqrt(eps_2_list(i));
    disp(eps_2_list(i))

    g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    ratios(i) = (g^2+eps^2)/g^2;

    means = 0;
    vars = 0;
    trials = 1;
    for j = 1:trials
        % get moments from simulated network
        W = get_W_twoblocks(N0, eps, g2total, conserved, "right blocks");
        temp = diag(R.^(-1/2)) - W;
        E = svd(temp).^(-2);
        means = means + mean(E);
        vars = vars + mean(E.^2);
    end
    % simulated statistics, trial averaged
    means_sim(i) = means/trials;
    var_sim(i) = vars/trials;

    % structured theory
    mu101 = a2/(1 - a2*g0^2);
    var_theory(i) = (a4* (16 + a2^2* eps^4 - a2^3 *eps^4 *g0^2 + 4* a2 *(eps^2 - 4 *g0^2)))/(4 *(-1 + a2* g0^2)^4 *(4 + a2* (eps^2 - 4 *g0^2)));

    % unstructured theory
    var_theory_ex2(i) = a4*(1/(1-a2*(g^2+eps^2/4))^4);
end

% figures
figure;
scatter(ratios, means_sim, 25, [0 0.4470 0.7410], "filled"); hold on
yline(mu101,"r",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('$\textsf{First moment }\varphi[C]$', 'FontSize', 16, 'Interpreter', 'latex');
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

figure;
scatter(ratios, var_sim, 25, [0 0.4470 0.7410], "filled"); hold on
plot(ratios, var_theory, "r",LineWidth=2);
plot(ratios, var_theory_ex2, "k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('$\textsf{Second moment }\varphi[C^2]$', 'FontSize', 16, 'Interpreter', 'latex');
xlim([1, ratios(end)])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
set(gcf,'position',[100,100,300,400])

%% Figure 8b (third panel). Effective dimension across within-area, between-area, and both within- and between-area structures

rng(42)
N0 = 2000;
N = 2*N0;
DR = 0.8; % D(R)
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; en
g2total = g0^2 * 4;
R = simr(targetMean, DR, N); % Generate firing rates
a2 = mean(R);
a4 = mean(R.^2);
DR = pr(R);

eps_2_list = 0:0.001:0.04;
ratios = zeros(length(eps_2_list),1); 
sim_cross = zeros(length(eps_2_list),1); 
sim_within = zeros(length(eps_2_list),1);
sim_both = zeros(length(eps_2_list),1);
theory_cross = zeros(length(eps_2_list),1); 
theory_within = zeros(length(eps_2_list),1); 
theory_both = zeros(length(eps_2_list),1); 
for i = 1:length(eps_2_list)
    eps = sqrt(eps_2_list(i));
    disp(eps_2_list(i))
    g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    ratios(i) = (g^2+eps^2)/g^2;

    sim_cross_trials = 0; 
    sim_within_trials = 0; 
    sim_both_trials = 0;
    trials = 10;
    for j = 1:trials
        W0 = randn(N)*g / sqrt(N);

        W_eps = [zeros(N0), zeros(N0); zeros(N0), randn(N0)*eps] / sqrt(N); % bottom right block
        W = W0 + W_eps;
        temp = diag(R.^(-1/2)) - W;
        E1 = svd(temp).^(-2);

        W_eps = [zeros(N0),randn(N0)*eps; zeros(N0), zeros(N0)] / sqrt(N); % top right block
        W = W0 + W_eps;
        temp = diag(R.^(-1/2)) - W;
        E2 = svd(temp).^(-2);

        W_eps = [zeros(N0),randn(N0)*eps/sqrt(2); zeros(N0),randn(N0)*eps/sqrt(2)] / sqrt(N); % both right blocks
        W = W0 + W_eps;
        temp = diag(R.^(-1/2)) - W;
        E3 = svd(temp).^(-2);

        sim_within_trials = sim_within_trials + pr(E1);
        sim_cross_trials = sim_cross_trials + pr(E2);
        sim_both_trials = sim_both_trials + pr(E3);
    end
    sim_cross(i) = sim_cross_trials/trials;
    sim_within(i) = sim_within_trials/trials;
    sim_both(i) = sim_both_trials/trials;
    
    % dc
    theory_cross(i) = (a2^2 * (1 + (eps^2 / 4) * a2)^2 * (1 - g^2 * a2 * (1 + (eps^2 / 4) * a2))^2) / (a4 * (1 + (eps^2 / 2) * a2)^2);
    theory_within(i) = DR*(1-a2*g^2+a2^2*eps^4/16)^2*(1-a2*(eps^2/4+g^2))^2/((1-a2*g^2)*(1-a2*g^2+eps^4/8*a2^2*(5-3*a2*g^2)));
    theory_both(i) = (4*DR*(-1 + a2*g0^2)^2* (-4 - a2 *eps^2 + 4 *a2* g0^2))/((-16 - a2^2 *eps^4 + a2^3 *eps^4 *g0^2 - 4 *a2* (eps^2 - 4 *g0^2)));
end
DC0 = DR * (1 - g0^2 * a2)^2;

figure;
scatter(ratios, sim_within, 25, "red", "filled","HandleVisibility","off"); hold on
scatter(ratios, sim_cross, 25, "blue", "filled","HandleVisibility","off");
scatter(ratios, sim_both, 25, [0.6902 0 1], "filled","HandleVisibility","off");
plot(ratios, theory_within, "r",LineWidth=2);
plot(ratios, theory_cross, "b",LineWidth=2);
plot(ratios, theory_both, color=[0.6902 0 1], LineWidth=2);
yline(DC0,"k",LineWidth=2);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex');
ylabel('$\textsf{Ef{}fective Dimension }D(C)$', 'FontSize', 16, 'Interpreter', 'latex');
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off; 
set(gcf,'position',[100,100,350,400])

%% Figure 8b (fourth panel). Symmetric variance structures do not change linear relationship

rng(42)
N0 = 500;
N = 2*N0;
nR = 50; % Number of conditions (i.e. operating points)
targetMean = 10; 
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;
DC = zeros(1, nR); DR = zeros(1, nR); % Initialize D(C) and D(R)

x = 0 : 0.1 : 1; 
y = x * (1 - (g^2+eps^2/4) * targetMean)^2; % Theoretical prediction

figure; hold on

eps = 0.29;
g = sqrt(g2total-eps^2)/2;
W0 = randn(N)*g / sqrt(N);

% within population
W_eps = [randn(N0)*eps/sqrt(2), zeros(N0); zeros(N0),randn(N0)*eps/sqrt(2)] / sqrt(N); 
W = W0 + W_eps;
for iR = 1 : nR
    targetPR = rand * 0.8 + 0.1; % Sample D(R) uniformly between 0.1 and 0.9
    [R, E] = simrc(W, targetMean, targetPR);
    DR(iR) = pr(R); DC(iR) = pr(E);
end

plot(DR, DC, '.r','MarkerSize',20); 

% cross population
W_eps = [zeros(N0), randn(N0)*eps/sqrt(2); randn(N0)*eps/sqrt(2), zeros(N0)] / sqrt(N); 
W = W0 + W_eps;
for iR = 1 : nR
    targetPR = rand * 0.8 + 0.1; % Sample D(R) uniformly between 0.1 and 0.9
    [R, E] = simrc(W, targetMean, targetPR);
    DR(iR) = pr(R); DC(iR) = pr(E);
end

plot(DR, DC, '.b','MarkerSize',20); 

plot(x, y,'k','LineWidth',3);  

legend('Within-area','Between-area','Theory','location','northwest');
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
box off;
legend boxoff;
set(gcf,'position',[100,100,300,400])
load('axes_position.mat', 'pos')
ax = gca;
set(ax, 'Position', pos)
xlabel('$\textsf{Firing rate uniformity }D(R)$', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$\textsf{Ef{}fective Dimension }D(C)$', 'FontSize', 15, 'Interpreter', 'latex');

%% Figure 9a. Eigenvalue distribution of area 1 cannot be obtained from Cauchy transform component G1 alone

rng(42)

% parameters
N0 = 2000;
N = 2*N0;
DR = 0.8; 
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;

x = linspace(0.1, 600, 3000);
R = simr(targetMean, DR, N); 

trials = 10;
E1s = zeros(N0,1); E2s = zeros(N0,1); Es = zeros(N,1);
for j = 1:trials
    % no added structure
    W = get_W_oneblock(N0, 0, g2total, conserved, "bottom right");
    % get simulated distribution of evals of C
    temp = diag(R.^(-1/2)) - W;
    C = (temp * temp') \ eye(size(temp));
    E1s = E1s + sort(svd(C(1:N0,1:N0)), 'descend');
    E2s = E2s + sort(svd(C(N0+1:end,N0+1:end)), 'descend');
    Es = Es + sort(svd(C), 'descend');
end
E1 = E1s/trials; 
E2 = E2s/trials; 
E = Es/trials;
[p1,p2] = computepdf_eps_Gs(x, R, g, eps);

f = figure; hold on;
plot(x,p1,'k-','LineWidth',2);
h1=histogram(E1,1000,'Normalization','pdf','DisplayStyle', 'stairs', 'LineWidth', 2,'Edgecolor', [0, 0.4470, 0.7410]);
legend("From G_1","Area 1 sims","location","northoutside","orientation","horizontal")
set(gca, 'XScale', 'log')
xlabel("Eigenvalue")
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,350,400])
box off;
legend box off;

%% Figure 9b. G1 is the projection of the Cauchy transform of T^{-2} onto the top left NxN block

rng(42)
N0 = 1000;
N = 2*N0;
DR = 0.8; % D(R)
targetMean = 10;

gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;

gridn = 1000;
x = logspace(0, 2.5, gridn);
z_imag = 0.01; % must be > 0 for upper half-plane
z_upper = x + 1i*z_imag;

R = simr(targetMean, DR, N); 
g = sqrt(g2total)/2;

trials = 30;
E1s = zeros(N0,1);
G_resolvent = zeros(size(x));
for j = 1:trials
    W = get_W_oneblock(N0, 0, g2total, conserved, "bottom right");
    % Get simulated distribution of evals of C
    temp = diag(R.^(-1/2)) - W;
    C = (temp' * temp) \ eye(size(temp));
    E1s = E1s + sort(svd(C(1:N0,1:N0)), 'descend');

    [V, D_eig] = eig(C);
    lambda = diag(D_eig);

    T_block = [(temp * temp') \ eye(size(temp)), zeros(N); zeros(N), C]; % 2N by 2N
    [V_full, D_full] = eig(T_block);
    lambda_full = diag(D_full);
    proj_A = sum(abs(V_full(1:N,:)).^2, 1);

    % projection of eigenmodes onto first block
    % proj_A = sum(abs(V(1:N0,:)).^2, 1);   % contribution weight per eigenvector, returns row after summing along columns
    for z=1:length(z_upper)
        for i=1:2*N
            G_resolvent(z) = G_resolvent(z) + (1/(N)) * 1/(z_upper(z)-lambda_full(i)) * proj_A(i);
        end
    end
end
G_resolvent = G_resolvent / trials;
[p1,p2] = computepdf_eps_Gs(x, R, g, 0);
p_weighted = -(1/pi) * imag(G_resolvent);

figure; hold on;
plot(x,p1,'k-','LineWidth',2); 
plot(x,p_weighted,"--r",'LineWidth',2.5); 
legend("From G_1","Weighted G_C","location","northoutside","orientation","horizontal")
set(gca, 'XScale', 'log')
xlabel("Eigenvalue")
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,350,400])
box off;
legend box off;

f = figure; hold on;
dz = diff(x);
dz = [dz, dz(end)]; 
c = cumsum(p_weighted.* dz); c = c / c(end);
c_old = cumsum(p1.* dz); c_old = c_old / c_old(end);
nRank = 40; 
ratio = 10; nRankT = (nRank-1) * ratio; 
Etheory_old = zeros(1, nRankT); Etheory = zeros(1, nRankT);
for iRank = 1 : nRankT
    Etheory_old(iRank) = x(find(c_old>1-(iRank+ratio)/ratio/N, 1));
    Etheory(iRank) = x(find(c>1-(iRank+ratio)/ratio/N, 1));
end
plot(1/ratio+1:1/ratio:nRank, Etheory_old,'k','LineWidth',2); hold on
plot(1/ratio+1:1/ratio:nRank, Etheory,'r--','LineWidth',2.5); hold on
xlabel('Rank'); ylabel('Eigenvalue')
set(gca,'xtick',[])
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,400,250])


%% Figure 10b. Linearization of area 1

rng(45)

% parameters
N0 = 2000;
N = 2*N0;
DR = 0.8; 
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;

% simulate network
R = simr(targetMean, DR, N); 
W = get_W_oneblock(N0, 0, g2total, true, "bottom right");
temp = diag(R.^(-1/2)) - W;
C = (temp * temp.') \ eye(size(temp));
E1 = eig(C(1:N0,1:N0));

% grid for evaluation (upper half-plane)
gridn = 500;
z_straight = logspace(0, 2.5, gridn);
epsilon = 0.05; % must be > 0 for upper half-plane
z_im = z_straight + 1i*epsilon;

% solve for theoretical density
G_p = computepdf_topleft(z_im,R,g0);

density_theory = -(1/pi) * imag(G_p);
density_theory(density_theory<0) = -density_theory(density_theory<0);

% plot eigenvalue distribution
figure;
histogram(E1, 1000, 'Normalization','pdf','DisplayStyle', 'stairs', 'LineWidth', 2,'Edgecolor','red','DisplayName', 'Area 1 Sims');

hold on;
plot(z_straight, density_theory, 'k-', 'LineWidth', 2, 'DisplayName', 'Linearization');
set(gca, 'XScale', 'log')
xlabel('Eigenvalue', 'FontSize', 14);
ylabel('Density', 'FontSize', 14);
legend('Location', 'northoutside','orientation','horizontal');
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,350,400])
box off;
legend box off;

% plot largest eigenvalues
figure;
% need to scale properly to handle log spaced grid 
dz = diff(z_straight);
dz = [dz, dz(end)];  % pad to same length
c = cumsum(density_theory .* dz);
c = c / c(end);
nRank = 40; 
Edata = sort(E1, 'descend'); Edata = Edata(1:nRank);
ratio = 10; nRankT = (nRank-1) * ratio; 

Etheory = zeros(1, nRankT);
for iRank = 1 : nRankT
    Etheory(iRank) = z_straight(find(c>1-(iRank+ratio)/ratio/N0, 1));
end
scatter(1:nRank, Edata, 'filled','red'); hold on
plot(1/ratio+1:1/ratio:nRank, Etheory,'k-','LineWidth',1.5); 
xlabel('Rank', 'FontSize', 14); 
set(gca,'xtick',[])
ylabel('Eigenvalue', 'FontSize', 14)
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,300,400,250])
box off;

%% Figure 10c. Dimension in area 1 vs the full network

rng(42)
N0 = 1000;
N = 2*N0;
DR = 0.8; 
targetMean = 10;
g0R = 0.01 : 0.01 : 0.99;
DPsim = zeros(length(g0R),1);
DCsim = zeros(length(g0R),1);
for i=1:length(g0R)
    disp(g0R(i))
    g0 = sqrt(g0R(i)/targetMean);
    g2total = 4*g0^2;
    eps = 0.0;
    R = simr(targetMean, DR, N); 
    W = get_W_oneblock(N0, eps, g2total, true, "bottom right");
    temp = diag(R.^(-1/2)) - W;
    C = (temp * temp.') \ eye(size(temp));
    E1 = eig(C(1:N0,1:N0));
    E = eig(C);
    DPsim(i)=pr(E1); DCsim(i)=pr(E);
end

DR = 0.8;
DP = DR*(1-g0R).^2*2./(1+(1-g0R).^2);
DC = DR*(1-g0R).^2;

figure;
plot(g0R,DP,"-r","Linewidth",2); hold on;
scatter(g0R,DPsim,20,"r","filled", 'HandleVisibility', 'off');
plot(g0R,DC,"-k","Linewidth",2);
scatter(g0R,DCsim,20,"k","filled", 'HandleVisibility', 'off');
xlabel('$g_0^2\varphi[R]$','Interpreter','latex'); ylabel('Effective dimension','fontsize',16);
legend("Area 1 D(P)","Full network D(C)");
legend box off
set(gca, 'FontSize', 16)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,350,400])
box off;
legend box off;

%% Figure S1. Eigenvalue distributions of W across different types of variance structures

rng(42)

% parameters
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;
eps = 0.3;
g = sqrt(g2total-eps^2)/2;
ang=0:0.01:2*pi;

N0 = 500;
N = 2*N0;
W0 = randn(N)*g / sqrt(N);
Ws = {randn(N)*g0 / sqrt(N), % circular law
    W0 + [zeros(N0), zeros(N0); zeros(N0), randn(N0)*eps] / sqrt(N), % bottom right block
    W0 + [zeros(N0), randn(N0)*eps; zeros(N0), zeros(N0)] / sqrt(N),  % top right block
    W0 + [zeros(N0), randn(N0)*eps/sqrt(2); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N), % both top and bottom right
    W0 + [randn(N0)*eps/sqrt(2), zeros(N0); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N), % both diagonals
    W0 + [zeros(N0), randn(N0)*eps/sqrt(2); randn(N0)*eps/sqrt(2), zeros(N0)] / sqrt(N)}; % both off diagonals

N0 = 2000;
N = 2*N0;
W0 = randn(N)*g / sqrt(N);
Ws_larger = {randn(N)*g0 / sqrt(N), % circular law
    W0 + [zeros(N0), zeros(N0); zeros(N0), randn(N0)*eps] / sqrt(N), % bottom right block
    W0 + [zeros(N0), randn(N0)*eps; zeros(N0), zeros(N0)] / sqrt(N),  % top right block
    W0 + [zeros(N0), randn(N0)*eps/sqrt(2); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N), % both top and bottom right
    W0 + [randn(N0)*eps/sqrt(2), zeros(N0); zeros(N0), randn(N0)*eps/sqrt(2)] / sqrt(N), % both diagonals
    W0 + [zeros(N0), randn(N0)*eps/sqrt(2); randn(N0)*eps/sqrt(2), zeros(N0)] / sqrt(N)}; % both off diagonals

% set up bins
n_bins = 20;
r_max = g0 * 1.5;
r_edges = linspace(0, r_max, n_bins + 1);
r_centers = (r_edges(1:end-1) + r_edges(2:end)) / 2;
dr = r_edges(2) - r_edges(1);

figure; 
for i=[1,2,3,7,8,9]
    ind = i;
    if i > length(Ws)
        ind = i-3;
    end
    W = Ws{ind};
    eigsW = eig(W);

    subplot(4,3,i); hold on;
    scatter(real(eigsW),imag(eigsW), 5 ,'filled','MarkerFaceColor',[.7 .7 .7]);
     
    xp=g0*cos(ang);
    yp=g0*sin(ang);
    plot(xp,yp,'k-',"Linewidth",1.5);

    xp=g*cos(ang);
    yp=g*sin(ang);
    plot(xp,yp,'k--',"Linewidth",1.5);
    xlabel('Real($\lambda$)','Interpreter','latex');
    if i == 1 || i == 7
        ylabel('Imag($\lambda$)','Interpreter','latex');
    end
    xlim([-0.25 0.25]);
    ylim([-0.25 0.25]);
    set(gca, 'FontSize', 10)
    set(gca, 'LineWidth', 2);
    box off;
    axis square

    W = Ws_larger{ind};
    eigsW = eig(W);
    r = abs(eigsW);

    density = zeros(1, n_bins);
    for k = 1:n_bins
        in_shell = r >= r_edges(k) & r < r_edges(k+1);
        shell_area = pi * (r_edges(k+1)^2 - r_edges(k)^2);
        density(k) = sum(in_shell) / (N * shell_area);
    end

    subplot(4,3,i+3); hold on;
    plot(r_centers, density, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
    xline(g0, 'k-', 'LabelVerticalAlignment', 'top', 'LineWidth', 1.5, 'Interpreter', 'tex');
    xline(g,  'k--','LabelVerticalAlignment', 'top', 'LineWidth', 1.5, 'Interpreter', 'tex');
    xlabel("Radius r")
    if i == 1 || i == 7
        ylabel("Density")
    end
    xlim([0, r_max]);
    set(gca, 'FontSize', 10, 'LineWidth', 2);
    box off;
    axis square
end

%% Figure S3b-c. Eigenvalue distributions with within-area structure and heterogeneous firing rate populations

rng(42)

% change target means of each area
targetMean1 = 6; 
targetMean2 = 7; 

N0 = 1000; 
N = N0*2;
gm = 1 / sqrt(max([targetMean1, targetMean2])); 
g0 = 0.6 * gm; 
g2total = g0^2 * 4;
DR = 0.8; 

% simulate firing rates
R1 = simr(targetMean1, DR, N0); 
R2 = simr(targetMean2, DR, N0); 
R = [R1; R2];

eps = sqrt(0.07);
g = sqrt(g2total-eps^2)/2;
x = linspace(0.1, 1000, 5000);
conserved = true;

trials = 10;
Es = zeros(N);
for j = 1:trials
    W = get_W_oneblock(N0, eps, g2total, conserved, "bottom right");
    % get simulated distribution of evals of C
    temp = diag(R.^(-1/2)) - W;
    E = svd(temp).^(-2);
    Es = Es + sort(E, 'descend');
end
E = Es/trials; 
p = computepdf_eps_rates(x, R1, R2, g, eps); % with structure, diff rates
p_oldeps = computepdf_eps(x, R, g, eps); % with structure, same rates
p_old = computepdf(x, R, g0); % no perturbation, same rates

% plot eigenvalue distribution
figure;
h=histogram(E,2000,'Normalization','pdf','HandleVisibility',"off"); hold on;
plot(x,p_old,'k','LineWidth',2);
plot(x,p_oldeps,'r','LineWidth',2);
plot(x,p,'b','LineWidth',2);
h.EdgeColor = 'none';
set(gca, 'XScale', 'log')
legend("Homo var","Homo rates","Hetero rates","Location","northoutside",'Orientation', 'horizontal')
xlabel("Eigenvalue")
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2);
box off;
legend boxoff;
set(gcf,'position',[100,100,500,300])

% get eigenvalues by rank
c = cumsum(p); c = c / c(end);
c_old = cumsum(p_old); c_old = c_old / c_old(end);
c_oldeps = cumsum(p_oldeps); c_oldeps = c_oldeps / c_oldeps(end);
nRank = 40; 
Edata = sort(E, 'descend'); Edata = Edata(1 : nRank); Etheory_oldeps = zeros(1, nRank);
ratio = 10; nRankT = nRank * ratio; 
Etheory_old = zeros(1, nRankT); Etheory = zeros(1, nRankT);
for iRank = 1 : nRankT
    Etheory_old(iRank) = x(find(c_old>1-iRank/ratio/N, 1));
    Etheory_oldeps(iRank) = x(find(c_oldeps>1-iRank/ratio/N, 1));
    Etheory(iRank) = x(find(c>1-iRank/ratio/N, 1));
end

% plot largest eigenvalues by rank
figure;
plot(1/ratio:1/ratio:nRank, Etheory_old,'k','LineWidth',2); hold on
scatter(1:nRank, Edata,36,[0 0.4470 0.7410], 'filled');
plot(1/ratio:1/ratio:nRank, Etheory,'b','LineWidth',2);  
plot(1/ratio:1/ratio:nRank, Etheory_oldeps, 'r', 'LineWidth',2);
xlabel('Rank'); ylabel('Eigenvalue')
set(gca,'xtick',[])
set(gca, 'FontSize', 14)
set(gca, 'LineWidth', 2);
set(gcf,'position',[100,100,300,200])
box off;

%% Figure S3f. Heatmaps of first and second partial derivatives of D(C) with respect to kappa^2

rng(42)
DR = 0.8;
g0 = 0.2268;
a12s = 4:0.02:12;
a22s = 4:0.02:12;

slopes = zeros(length(a12s), length(a22s));
secondslopes = zeros(length(a12s), length(a22s));
zero_slopes = zeros(length(a12s));
zero_secondslopes = NaN(length(a12s));
zero_secondslopes2 = NaN(length(a12s));

% calculate partials
for i=1:length(a12s)
    for j=1:length(a22s)
        a22 = a22s(j);
        a12 = a12s(i);
        a14 = a12^2/DR;
        a24 = a22^2/DR;
        slopes(i,j) = -(1/(16* (a14 + a24)^2))* (a12 + a22)* (-2 + a12* g0^2 + a22 *g0^2) *(a12^2 *(a14 + a24 + 4 *a22* a24 *g0^2) + a22^2* (-3 *a24 + a14* (5 - 4 *a22* g0^2)) + 2* a12 *a22* (a14 - 2 *a14* a22 *g0^2 + a24 *(-3 + 2* a22 *g0^2)));
        secondslopes(i,j) = 1/(128*(a14 + a24)^3) *( 8*a12^2*a22^2*(a14 + a24)^2*(-2 + a12*g0^2 + a22*g0^2)^2 ...
  - a12*a22*(a12 + a22)*(a14 + a24)*(4 - 2*(a12 + a22)*g0^2) * ...
    (8*(a14 + a24)*(a12 - 3*a22 + 2*a12*a22*g0^2) + 16*a14*a22*(4 - 2*(a12 + a22)*g0^2)) ...
  + 4*(a12 + a22)^2 * ( 4*a14*a22^2*(5*a14 - 3*a24)*(-2 + a12*g0^2 + a22*g0^2)^2 ...
      - 8*a14*a22*(a14 + a24)*(-2 + a12*g0^2 + a22*g0^2)*(a12 - 3*a22 + 2*a12*a22*g0^2) ...
      - (a14 + a24)^2*( -(1/2)*(a12 - 3*a22 + 2*a12*a22*g0^2)^2 + a12*a22*(4 - 2*(a12 + a22)*g0^2) )));
    end
end

% plot partials
darkToLightBlue = @(n) [linspace(0.02,0.75,n)', linspace(0.02,0.85,n)', linspace(0.35,1,n)'];
figure('position', [100, 100, 900, 300]);
subplot(1,2,1)
imagesc(a12s, a22s, slopes');
set(gca, 'YDir', 'normal');
set(gca, 'FontSize', 14);
hold on;
contour(a12s, a22s, slopes', [0 0], 'r-', 'LineWidth', 2);
plot(6,7, 'cp','MarkerSize', 10,'MarkerFaceColor','c');
plot(7,6, 'yp','MarkerSize', 10,'MarkerFaceColor','y');
xlabel('$\varphi[R_1]$','Interpreter','latex','FontSize', 18);
ylabel('$\varphi[R_2]$','Interpreter','latex','FontSize', 18);
title('$\frac{\partial D(C)}{ \partial \kappa^2} \bigg|_{\kappa^2=0}$','Interpreter','latex','FontSize', 18);
colormap(gca, darkToLightBlue(length(a12s)));
cb = colorbar('eastoutside'); cb.LineWidth = 2;
set(gca, 'LineWidth', 2);

subplot(1,2,2)
imagesc(a12s, a22s, secondslopes');
set(gca, 'YDir', 'normal');
set(gca, 'FontSize', 14);
hold on;
contour(a12s, a22s, secondslopes', [0 0], 'r-', 'LineWidth', 2);
plot(6,7, 'cp','MarkerSize', 10,'MarkerFaceColor','c');
plot(7,6, 'yp','MarkerSize', 10,'MarkerFaceColor','y');
xlabel('$\varphi[R_1]$','Interpreter','latex','FontSize', 18);
ylabel('$\varphi[R_2]$','Interpreter','latex','FontSize', 18);
title('$\frac{\partial^2 D(C)}{ \partial (\kappa^2)^2} \bigg|_{\kappa^2=0}$','Interpreter','latex','FontSize', 18);
colormap(gca, darkToLightBlue(length(a12s)));
cb = colorbar('eastoutside'); cb.LineWidth = 2;
set(gca, 'LineWidth', 2);


%% Figure S4. Moments and dimension for each area and the full covariance matrix

rng(42)

% parameters
N0 = 1000;
N = 2*N0;
DR = 0.8; % D(R)
targetMean = 10;
gm = 1 / sqrt(targetMean); 
g0 = 0.6 * gm; 
g2total = 4*g0^2;

trials = 20;
conserved = true;

eps_2_list = 0:0.002:0.03;
E1s = zeros(size(eps_2_list));E2s = zeros(size(eps_2_list));
E21s = zeros(size(eps_2_list));E22s = zeros(size(eps_2_list));
DC1s = zeros(size(eps_2_list));DC2s = zeros(size(eps_2_list));
Efulls = zeros(size(eps_2_list));E2fulls = zeros(size(eps_2_list));DCfulls=zeros(size(eps_2_list));
ratios = zeros(size(eps_2_list));
for i = 1:length(eps_2_list)
    eps = sqrt(eps_2_list(i));
    disp(eps_2_list(i))
    g = sqrt(g2total-eps^2)/2; % g^2 = g0^2-eps^2/4, variance conserved
    ratios(i) = (g^2+eps^2)/g^2;

    E1 = 0; E2 = 0;
    E21 = 0; E22 = 0;
    DC1 = 0; DC2 = 0;
    Efull = 0; E2full = 0; DCfull = 0;
    for j = 1:trials
        R = simr(targetMean, DR, N); 

        % change this for different structured cases
        % W = get_W_oneblock(N0, eps, g2total, conserved, "top right");
        W = get_W_twoblocks(N0, eps, g2total, conserved, "off diagonals");

        % get simulated distribution of evals of C
        temp = diag(R.^(-1/2)) - W;
        C = (temp' * temp) \ eye(size(temp));
        C1 = C(1:N0, 1:N0);
        C2 = C(N0+1:end, N0+1:end);
        eig1 = eig(C1); eig2 = eig(C2); eigfull = eig(C);

        % eigenvalues of area 1
        E1 = E1 + mean(eig1);
        E21 = E21 + mean(eig1.^2);
        DC1 = DC1 + mean(eig1)^2/mean(eig1.^2);

        % eigenvalues of area 2
        E2 = E2 + mean(eig2);
        E22 = E22 + mean(eig2.^2);
        DC2 = DC2 + mean(eig2)^2/mean(eig2.^2);

        % eigenvalues of full covariance matrix
        Efull = Efull + mean(eigfull);
        E2full = E2full + mean(eigfull.^2);
        DCfull = DCfull + mean(eigfull)^2/mean(eigfull.^2);
    end

    E1s(i) = E1/trials; E2s(i) = E2/trials;
    E21s(i) = E21/trials; E22s(i) = E22/trials;
    DC1s(i) = DC1/trials; DC2s(i) = DC2/trials;
    Efulls(i) = Efull/trials; E2fulls(i) = E2full/trials; DCfulls(i) = DCfull/trials;
end

figure;
t = tiledlayout(1, 3); 
set(gcf,'position',[100,300,1000,350])

nexttile
p1 = plot(ratios,E1s,"-r","Linewidth",2); hold on;
p2 = plot(ratios,E2s,"-b","Linewidth",2);
p3 = plot(ratios,Efulls,"-k","Linewidth",2);
plot(ratios,E1s,"r.","Markersize",15);
plot(ratios,E2s,"b.","Markersize",15);
plot(ratios,Efulls,"k.","Markersize",15);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex'); ylabel('First moment','fontsize',16);
set(gca, 'FontSize', 16); set(gca, 'LineWidth', 2); box off;
ylim([14,17])

nexttile
plot(ratios,E21s,"-r","Linewidth",2); hold on;
plot(ratios,E22s,"-b","Linewidth",2);
plot(ratios,E2fulls,"-k","Linewidth",2);
plot(ratios,E21s,"r.","Markersize",15);
plot(ratios,E22s,"b.","Markersize",15);
plot(ratios,E2fulls,"k.","Markersize",15);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex'); ylabel('Second moment','fontsize',16);
set(gca, 'FontSize', 16); set(gca, 'LineWidth', 2); box off;
ylim([0.95*min([E22s,E2fulls,E21s]),1.05*max([E22s,E2fulls,E21s])])

nexttile
plot(ratios,DC1s,"-r","Linewidth",2); hold on;
plot(ratios,DC2s,"-b","Linewidth",2);
plot(ratios,DCfulls,"-k","Linewidth",2);
plot(ratios,DC1s,"r.","Markersize",15);
plot(ratios,DC2s,"b.","Markersize",15);
plot(ratios,DCfulls,"k.","Markersize",15);
xlabel('$(g^2+\kappa^2)/g^2$','Interpreter','latex'); ylabel('D(C)','fontsize',16);
set(gca, 'FontSize', 16); set(gca, 'LineWidth', 2); box off;

lgd = legend([p1, p2, p3], {'Area 1', 'Area 2', "Full covariance matrix"});
lgd.Layout.Tile = 'north';
lgd.Orientation = 'horizontal';
lgd.Box = 'off';
