import asyncio
import pandas as pd
from data.fetcher import get_fetcher, BaseFetcher
from data.storage.sqlite_adapter import SQLiteAdapter
from data.manager import DataManager
from data import features as feat

print('fetcher factory default:', type(get_fetcher()))
print('fetcher factory polygon class available:', get_fetcher('polygon').__class__.__name__)
print('fetcher factory alpha class available:', get_fetcher('alpha_vantage').__class__.__name__)

# test sqlite in-memory
storage = SQLiteAdapter(db_path=':memory:')
print('sqlite in-memory created, total rows:', storage.get_total_price_count())

# create dummy fetcher to avoid network
class DummyFetcher(BaseFetcher):
    async def fetch(self, ticker, start, end, interval):
        import pandas as pd
        return pd.DataFrame()

# instantiate manager with dummy fetcher
mgr = DataManager(storage, fetcher=DummyFetcher())
res = asyncio.run(mgr.fetch(['SPY'], interval='1d', refresh=True))
print('fetch with dummy returned rows:', len(res))

# test feature functions on small series
s = pd.Series([1,2,3,4,5,6,7,8,9,10], name='close')
print('rsi len:', len(feat.rsi(s)))
print('macd columns:', feat.macd(pd.DataFrame({'close': s})).columns.tolist())
print('bollinger columns:', feat.bollinger_bands(pd.DataFrame({'close': s})).columns.tolist())
print('done')
