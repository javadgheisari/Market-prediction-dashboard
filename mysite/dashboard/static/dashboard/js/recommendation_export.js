(function () {
    const button = document.querySelector('[data-export-ranking]');
    const dataElement = document.getElementById('recommendation-export-data');
    if (!button || !dataElement) return;

    const recommendations = JSON.parse(dataElement.textContent);
    const mode = button.dataset.exportMode;
    const columns = mode === 'model_ensemble'
        ? [
            ['rank', 'Rank'],
            ['symbol', 'Asset'],
            ['decision', 'Recommendation'],
            ['ranking_score', 'Ensemble score'],
            ['xgboost_score', 'XGBoost score'],
            ['gru_score', 'GRU score'],
            ['lstm_score', 'LSTM score'],
            ['positive_components', 'Positive models'],
            ['component_count', 'Model count'],
            ['expected_return_percent', 'Mean forecast return (%)'],
            ['confidence_percent', 'Mean confidence (%)'],
            ['current_price', 'Current price'],
            ['target_price', 'Mean forecast target'],
            ['price_source', 'Price source'],
            ['news_enabled', 'News features included'],
            ['forecast_is_naive', 'All price forecasts naive'],
            ['error', 'Error'],
        ]
        : mode === 'ensemble'
        ? [
            ['rank', 'Rank'],
            ['symbol', 'Asset'],
            ['decision', 'Recommendation'],
            ['ranking_score', 'Ensemble score'],
            ['xgboost_score', 'XGBoost score'],
            ['gru_score', 'GRU score'],
            ['lstm_score', 'LSTM score'],
            ['news_score', 'News score'],
            ['news_signal_score', 'Raw news signal'],
            ['positive_components', 'Positive components'],
            ['component_count', 'Component count'],
            ['expected_return_percent', 'Mean forecast return (%)'],
            ['confidence_percent', 'Mean confidence (%)'],
            ['current_price', 'Current price'],
            ['target_price', 'Mean forecast target'],
            ['price_source', 'Price source'],
            ['forecast_is_naive', 'All price forecasts naive'],
            ['error', 'Error'],
        ]
        : mode === 'forecast'
        ? [
            ['rank', 'Rank'],
            ['symbol', 'Asset'],
            ['decision', 'Recommendation'],
            ['ranking_score', 'Ranking score'],
            ['expected_return_percent', 'Forecast return (%)'],
            ['bullish_percent', 'Bullish probability (%)'],
            ['confidence_percent', 'Confidence (%)'],
            ['directional_accuracy_percent', 'Directional accuracy (%)'],
            ['current_price', 'Current price'],
            ['target_price', 'Forecast target'],
            ['price_source', 'Price source'],
            ['news_enabled', 'News included'],
            ['forecast_is_naive', 'Naive forecast'],
            ['error', 'Error'],
        ]
        : [
            ['rank', 'Rank'],
            ['symbol', 'Asset'],
            ['decision', 'Recommendation'],
            ['ranking_score', 'Ranking score'],
            ['signal_score', 'News signal'],
            ['confidence_percent', 'Confidence (%)'],
            ['articles_analyzed', 'Active articles'],
            ['articles_saved', 'Saved articles'],
            ['fetched', 'Articles fetched'],
            ['error', 'Error'],
        ];

    function csvCell(value) {
        if (value === null || value === undefined) return '';
        let text = String(value);
        if (typeof value === 'string' && /^[=+\-@]/.test(text)) text = `'${text}`;
        return `"${text.replace(/"/g, '""')}"`;
    }

    button.addEventListener('click', function () {
        const rows = [
            columns.map(function (column) { return csvCell(column[1]); }),
            ...recommendations.map(function (item) {
                return columns.map(function (column) { return csvCell(item[column[0]]); });
            }),
        ];
        const csv = rows.map(function (row) { return row.join(','); }).join('\r\n');
        const blob = new Blob([`\uFEFF${csv}`], {type: 'text/csv;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = button.dataset.exportFilename || 'buy-ranking.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    });
})();
