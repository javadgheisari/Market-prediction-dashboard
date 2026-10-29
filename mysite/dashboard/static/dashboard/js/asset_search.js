(function () {
    const input = document.getElementById('asset-search-input');
    const clearButton = document.getElementById('asset-search-clear');
    const status = document.getElementById('asset-search-status');
    const empty = document.getElementById('asset-search-empty');
    if (!input || !clearButton || !status || !empty) return;

    const groups = Array.from(document.querySelectorAll('[data-market-category]'));
    const total = groups.reduce((count, group) => count + group.querySelectorAll('[data-asset-symbol]').length, 0);

    function filterAssets() {
        const query = input.value.trim().toLocaleLowerCase();
        let visible = 0;
        groups.forEach(function (group) {
            const categoryMatches = group.dataset.marketCategory.toLocaleLowerCase().includes(query);
            let groupVisible = 0;
            group.querySelectorAll('[data-asset-symbol]').forEach(function (row) {
                row.hidden = Boolean(query) && !categoryMatches && !row.dataset.assetSymbol.toLocaleLowerCase().includes(query);
                if (!row.hidden) groupVisible += 1;
            });
            group.hidden = groupVisible === 0;
            visible += groupVisible;
        });
        clearButton.hidden = !query;
        empty.hidden = visible !== 0;
        status.textContent = query
            ? `${visible} matching asset${visible === 1 ? '' : 's'}`
            : `${total} assets available`;
    }

    input.addEventListener('input', filterAssets);
    clearButton.addEventListener('click', function () {
        input.value = '';
        filterAssets();
        input.focus();
    });
    filterAssets();
})();
