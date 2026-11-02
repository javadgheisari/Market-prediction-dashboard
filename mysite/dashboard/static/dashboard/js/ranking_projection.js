(function () {
    const dataNode = document.getElementById('ranking-projection-data');
    const chart = document.getElementById('ranking-projection-chart');
    const assetSelect = document.getElementById('ranking-projection-asset');
    const summary = document.getElementById('ranking-projection-summary');
    if (!dataNode || !chart || !assetSelect || !summary) return;

    if (typeof Plotly === 'undefined') {
        chart.textContent = 'The chart library could not be loaded. Refresh the page to try again.';
        return;
    }

    const items = JSON.parse(dataNode.textContent);
    const labels = {xgboost: 'XGBoost', gru: 'GRU', lstm: 'LSTM', chronos: 'Chronos-2'};
    const colors = {xgboost: '#60a5fa', gru: '#34d399', lstm: '#c084fc', chronos: '#60a5fa'};

    function render() {
        const item = items[Number(assetSelect.value)];
        if (!item) return;

        const paths = item.projection.paths;
        const modelNames = Object.keys(paths);
        const aggregate = item.projection.aggregate;
        const steps = aggregate.map((_, index) => index);
        const multipleModels = modelNames.length > 1;
        const traces = multipleModels ? modelNames.map((name) => ({
            type: 'scatter',
            mode: 'lines',
            name: labels[name] || name,
            x: steps,
            y: paths[name],
            line: {color: colors[name] || '#60a5fa', width: 2, dash: 'dot'},
            hovertemplate: '%{fullData.name}: %{y:+.2f}%<extra></extra>'
        })) : [];
        traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: multipleModels ? 'Aggregate mean' : `${labels[modelNames[0]] || modelNames[0]} projection`,
            x: steps,
            y: aggregate,
            line: {color: '#fb923c', width: 3},
            marker: {color: '#fb923c', size: 5},
            hovertemplate: '%{fullData.name}: %{y:+.2f}%<extra></extra>'
        });

        const finalMove = aggregate[aggregate.length - 1];
        summary.textContent = `${item.symbol}: ${multipleModels ? 'mean' : 'model'} projects ${finalMove >= 0 ? '+' : ''}${finalMove.toFixed(2)}% after ${steps.length - 1} forecast candles.`;
        chart.setAttribute('aria-label', `${item.symbol} projected movement: ${finalMove >= 0 ? 'up' : 'down'} ${Math.abs(finalMove).toFixed(2)} percent after ${steps.length - 1} candles`);

        Plotly.react(chart, traces, {
            paper_bgcolor: '#0b1728',
            plot_bgcolor: '#0f1d30',
            font: {color: '#cbd5e1', family: 'DM Sans, sans-serif'},
            margin: {l: 65, r: 24, t: 36, b: 58},
            xaxis: {title: 'Forecast candle', gridcolor: 'rgba(148,163,184,.16)', zeroline: false, dtick: Math.max(1, Math.ceil((steps.length - 1) / 8))},
            yaxis: {title: 'Move from latest price (%)', gridcolor: 'rgba(148,163,184,.16)', zerolinecolor: 'rgba(148,163,184,.4)', ticksuffix: '%'},
            legend: {orientation: 'h', y: 1.16},
            hovermode: 'x unified'
        }, {responsive: true, displaylogo: false});
    }

    assetSelect.addEventListener('change', render);
    render();
}());
