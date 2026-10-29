(function () {
    const button = document.getElementById('forecast-csv-download');
    const dataElement = document.getElementById('chart-data');
    if (!button || !dataElement) return;

    function csvCell(value) {
        if (value === null || value === undefined) return '';
        let text = String(value);
        if (typeof value === 'string' && /^[\s]*[=+\-@]/.test(text)) text = "'" + text;
        return '"' + text.replace(/"/g, '""') + '"';
    }

    function forecastCsv(data) {
        const interval = data.forecast_interval_percent;
        const boundLabel = interval == null ? 'Forecast' : `${interval}% forecast`;
        const rows = [[
            'Series', `Date and time (${data.timezone || 'Asia/Tehran'})`,
            'Actual price', 'Test prediction', 'Future prediction',
            `${boundLabel} lower`, 'Forecast median', `${boundLabel} upper`
        ]];
        (data.actual_dates || []).forEach(function (date, index) {
            rows.push(['Actual', date, (data.actual_prices || [])[index], '', '', '', '', '']);
        });
        (data.test_dates || []).forEach(function (date, index) {
            rows.push(['Test', date, '', (data.test_predictions || [])[index], '', '', '', '']);
        });
        (data.future_dates || []).forEach(function (date, index) {
            rows.push([
                'Future', date, '', '', (data.future_predictions || [])[index],
                (data.future_lower || [])[index], (data.future_median || [])[index],
                (data.future_upper || [])[index]
            ]);
        });
        return '\ufeff' + rows.map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
    }

    button.addEventListener('click', function () {
        const data = JSON.parse(dataElement.textContent);
        const url = URL.createObjectURL(new Blob([forecastCsv(data)], {type: 'text/csv;charset=utf-8'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = button.dataset.exportFilename || 'forecast.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    });
})();
