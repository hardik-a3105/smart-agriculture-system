const dns = require('dns');

const domain = '_mongodb._tcp.cluster0.t0tlop2.mongodb.net';

console.log(`Resolving SRV record for: ${domain}`);

dns.resolveSrv(domain, (err, addresses) => {
    if (err) {
        console.error('❌ DNS Resolution Failed:', err);
    } else {
        console.log('✅ DNS Resolution Successful:');
        console.log(addresses);
    }
});
