## Data sources and methodology

Daily on-chain transaction volume is calculated as the sum of all transaction outputs belonging to the blocks mined on the given day. “Change” outputs are not included. Estimation difficulties remain and the measure is imprecise. We discuss this here. Methodology behind adjusted transaction volume figures is described in [this post](https://coinmetrics.io/introducing-adjusted-estimates/).

Transaction count figure doesn't include coinbase and coinstake transactions.

Active addresses is the number of unique sending and receiving addresses participating in transactions on the given day.

Payment count for UTXO coins is defined as sum of outputs’ count minus one for each transaction. We assume that transaction with N outputs pays to N – 1 addresses and the last N-th output is change. Transactions with only one output do not contribute to payment count, as they are likely to be a self-churn.
Payment count for smart contract assets such as ETH or LSK is calculated as the amount of transfer transactions (i.e. contract creation, invocation, destruction transactions are not included).

NEO and GAS transaction count figures reflect the amount of transactions that have at least one output of given asset type. If transaction sends both NEO and GAS, it will be included in transaction count for both assets. Fees figure is denominated in GAS and calculated by summing the fees of all transactions that have at least one output of a given asset type.

Ripple data includes only transactions of Payment type that transfer XRP tokens.

Stellar transaction volume data covers only operations of Payment and CreateAccount types that transfer XLM tokens. Transaction count is the number of transactions that include at least one operation of aforementioned types. Lumens inflation data is currently unavailable.

XEM data includes only transactions of “Transfer” type.

Zcash figures for on-chain volume and transaction count reflect data collected for transparent transactions only. In the last month, 9.1% (14/06/18) of ZEC transactions were shielded, and these are excluded from the analysis due to their private nature. Thus transaction volume figures in reality are higher than the estimate presented here, and NVT and exchange to transaction value lower. Data on shielded and transparent transactions can be found here and here.

Monero transaction volume is impossible to calculate due to RingCT which hides transaction amounts.

EOS transaction volume figure includes only transactions of transfer type. Median transaction value for EOS is actually median transfer value (currently, most transactions in EOS do not transfer EOS token).

WAVES transaction volume figure includes only transactions of transfer and mass transfer types. Median transaction value for WAVES is actually median value of WAVES token transfer.

Price data
All coins: coinmarketcap.com

On-chain transaction volume data
BTC, BCH, LTC, DCR, DASH, ZEC, DOGE, PIVX, XVG, VTC, DGB, BTG, USDT, MAID: data collected from blockchains and aggregated by CM Python tools
ETH and ERC20 tokens, ETC, XMR, XEM, ADA, LSK, NEO, GAS: data collected from blockchains by CM Haskell tools and aggregated by companion analytics scripts
XRP: data collected from data.ripple.com by CM Haskell tools and aggregated by companion analytics scripts
XLM: data collected from history.stellar.org by CM Haskell tools and aggregated by companion analytics scripts
Coinmetrics is a data aggregator and visualization service. Any written content is opinion, and should be used for informational purposes only. Content on this website is not intended to and does not constitute investment advice. Seek a licensed professional for investment advice.


## Fields

**txVolume(USD)** - on-chain transaction volume. A broad and largely unadjusted measure of the total value of outputs on the blockchain, on a given day. This is an answer to the question “approximately how much value, denominated in USD, circulates on the Bitcoin blockchain a day?”

That said, on-chain transaction volume in practice is a very hard thing to estimate properly. We discuss that at length in this post. Please read it! We don’t want to give anyone a false degree of certainty in the number. We are currently working through various methods to improve the estimate, and maybe get closer to blockchain.info’s adjusted figure – but that will take time. So bear with us.

**txCount** - refers to the number of transactions happening on the public blockchain a day. Be aware that for low-fee blockchains, it’s really easy to fabricate a whole bunch of transactions. So this isn’t necessarily that reliable! Additionally, UTXO networks like Bitcoin can batch a whole bunch of transactions into one, so txCount underestimates those ones. You have to therefore be careful comparing the number of transactions on Ethereum with Bitcoin; by its very nature, Bitcoin typically has more transactions than that datapoint suggests. Naively comparing the UTXO to account based systems by transaction count is like watching motorway traffic and comparing the number of buses versus motorcycles to guess at how many people are making trips. Maybe there are the same number of buses and motorcycles – but each bus might have 50 people inside of it.

Here’s a very typical example of a batched Bitcoin transaction. One sender, lots of receivers.

![bitcoin-transaction](https://coinmetrics.io/wp-content/uploads/2018/01/Screen-Shot-2018-01-28-at-7.13.12-PM-768x436.png)

This is fairly typical for Bitcoin. Maybe it’s a mining pool paying out, or an exchange paying multiple users at once. Batching transactions saves space, and is more cost-efficient, so it’s encouraged. It also means that just counting transactions for Bitcoin (or other UTXO chains) isn't likely to yield a reliable estimate of how many actual transactions are occurring. The useful site outputs.today popped up recently to make this point. Of course, outputs.today might be including change outputs (we’re not sure if they are or not), which aren't meaningful transfers. But even if you conservatively assumed that half of all bitcoin outputs tallied on outputs.today are change outputs, you would still have a higher number of outputs than just the raw number of transactions. So please be aware of the fact that Bitcoin transactions have the flexibility of email (one can send to many), constrained only by the blocksize and the willingness of miners to include large transactions.

**marketcap(USD)** - This is the unit price multiplied by the number of units in circulation. There has been quite a bit of controversy over this indicator. We still like this post from the Sia guys on the topic. Marketcap or network value is definitely flawed. It becomes less tethered to reality the smaller the float is. Float means the ratio of actual circulating units to the total number of units. Ripple, for instance, has a fairly small float, so one should probably be skeptical of its “market cap.” OnChainFx is doing a lot of good work on the issue.

**Price** - not much to say about this one. We get it from CoinMarketCap, with all the caveats that entails. Be advised – it’s the opening price.

**exchangevolume(USD)** is, as you might expect, the dollar value of the volume at exchanges like GDAX and Bitfinex. We get this data to Coinmarketcap, who have a bit of a conflicted history with the figure, having deleted and re-added Korean exchange figures. It doesn’t include data on OTC exchanges, which is a meaningful portion of all global exchange. Remember that 0-fee exchange volume should be taken with a grain of salt.

**generatedCoins** - this refers to the number of new coins that have been brought into existence on that day. We count up the actual number of newly-minted coins, rather than using the stated inflation figures (i.e. for bitcoin you should expect 12.5 per block, every ten minutes, giving you 12.5*6*24 = 1800 coins per day). You can see that we’ve been exceeding 1800 coins per day recently – this is due to lots of new hashpower which is coming online and making blocks come every 8 or 9 minutes than 10. In practice, since hashpower is continually added to the system, Bitcoin inflation progresses slightly faster than its theoretical rate. This is also why our figures differ from those of other websites – we count up the actual number of new coins rather than just assuming the official inflation rates are correct.

**Fees** - Fees in our data are based on the native currency, not USD. So on January 28th, fees totaled 168.25 BTC. That’s about $1.88m. This has been a source of confusion for many, so again – fees are counted in the native currency. You have to multiply by unit price to obtain the USD value of fees. Fees are interesting because they can’t really be faked – either you want to use the blockchain, and you’ll pay to do it, or you choose not to.

**averaDifficulty** - average measure of how difficult it is to find a hash below a given target. The Bitcoin network has a global block difficulty. Valid blocks must have a hash below this target. Mining pools also have a pool-specific share difficulty setting a lower limit for shares.

So to conclude, on-chain volume and transaction count can both be faked and can be tricky to estimate. Exchange volume must be viewed fairly skeptically. Market cap has a whole host of methodological issues. Generated coins and fees, however, are much more concrete.

We hope this helps you assess our data. Do not take it as gospel – use it only with the appropriate dose of skepticism.

