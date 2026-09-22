function visualizeLatentAndWaveforms(trainedNet, inputData, latentLayer, selIdx)
% VISUALIZELATENTANDWAVEFORMS
%   trainedNet  : trained dlnetwork or DAGNetwork
%   inputData   : input data [N x inpsize] (seismic patches)
%   latentLayer : name of the layer to extract (e.g., 'fc7' or 'temp_scaling')
%   selIdx      : indices of samples to highlight and plot waveforms
%
% Example:
%   visualizeLatentAndWaveforms(trainedNet, inputData, 'fc7', [10,200,500])

    % Extract latent activations
    act_latent = activations(trainedNet, inputData, latentLayer, 'OutputAs', 'rows');
    
    % Perform PCA
    [coeff, score, ~] = pca(act_latent);

    % Scatter plot of first 2 PCs
    figure;
    scatter(score(:,1), score(:,2), 15, 'filled');
    hold on;
    plot(score(selIdx,1), score(selIdx,2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
    title(['PCA of Latent Layer: ', latentLayer]);
    xlabel('PC1'); ylabel('PC2');
    legend('All Samples', 'Selected', 'Location', 'best');
    grid on;

    % Plot corresponding waveforms
    figure;
    for i = 1:length(selIdx)
        subplot(length(selIdx),1,i);
        plot(inputData(selIdx(i),:));
        title(sprintf('Waveform of Sample %d (linked to latent)', selIdx(i)));
    end

end
