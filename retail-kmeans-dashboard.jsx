import React, { useState, useMemo, useCallback } from 'react';
import Papa from 'papaparse';
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis,
} from 'recharts';

/* ============================================================
   TOKENS
   ============================================================ */
const C = {
  bg: '#11151C',
  surface: '#1A2029',
  surfaceAlt: '#222B38',
  border: '#2E3848',
  amber: '#E3A23C',
  teal: '#4FD1C5',
  rose: '#E8765C',
  violet: '#9D8DF1',
  green: '#7FD88F',
  blue: '#6B9BD1',
  text: '#F2EFE9',
  muted: '#8C96A8',
};
const CLUSTER_COLORS = [C.amber, C.teal, C.rose, C.violet, C.green, C.blue, '#D1A3D8', '#F2C572'];

const fmtIDR = (v) =>
  new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(v || 0);
const fmtNum = (v, d = 0) => new Intl.NumberFormat('id-ID', { maximumFractionDigits: d }).format(v || 0);

/* ============================================================
   DATE PARSING
   ============================================================ */
function parseDate(val) {
  if (val instanceof Date) return isNaN(val.getTime()) ? null : val;
  if (val == null || val === '') return null;
  const s = String(val).trim();
  let d = new Date(s);
  if (!isNaN(d.getTime())) return d;
  const m = s.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})/);
  if (m) {
    let [, a, b, c] = m;
    if (c.length === 2) c = '20' + c;
    d = new Date(`${c}-${b.padStart(2, '0')}-${a.padStart(2, '0')}`);
    if (!isNaN(d.getTime())) return d;
  }
  return null;
}

/* ============================================================
   K-MEANS (pure JS implementation)
   ============================================================ */
function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function initCentroidsPlusPlus(data, k) {
  const centroids = [data[Math.floor(Math.random() * data.length)]];
  while (centroids.length < k) {
    const dists = data.map((p) => {
      let min = Infinity;
      for (const c of centroids) {
        const d = euclideanDist(p, c);
        if (d < min) min = d;
      }
      return min * min;
    });
    const sum = dists.reduce((a, b) => a + b, 0);
    if (sum === 0) {
      centroids.push(data[Math.floor(Math.random() * data.length)]);
      continue;
    }
    let r = Math.random() * sum;
    let acc = 0;
    let chosen = data[0];
    for (let i = 0; i < data.length; i++) {
      acc += dists[i];
      if (acc >= r) {
        chosen = data[i];
        break;
      }
    }
    centroids.push(chosen);
  }
  return centroids.map((c) => [...c]);
}

function runKMeans(data, k, maxIter = 50, restarts = 4) {
  if (!data.length) return { assignments: [], centroids: [], wcss: 0 };
  const effK = Math.max(1, Math.min(k, data.length));
  let best = null;
  for (let r = 0; r < restarts; r++) {
    let centroids = initCentroidsPlusPlus(data, effK);
    let assignments = new Array(data.length).fill(0);
    for (let iter = 0; iter < maxIter; iter++) {
      let changed = false;
      for (let i = 0; i < data.length; i++) {
        let bestDist = Infinity;
        let bestC = 0;
        for (let c = 0; c < effK; c++) {
          const d = euclideanDist(data[i], centroids[c]);
          if (d < bestDist) {
            bestDist = d;
            bestC = c;
          }
        }
        if (assignments[i] !== bestC) {
          assignments[i] = bestC;
          changed = true;
        }
      }
      const dim = data[0].length;
      const sums = Array.from({ length: effK }, () => new Array(dim).fill(0));
      const counts = new Array(effK).fill(0);
      for (let i = 0; i < data.length; i++) {
        counts[assignments[i]]++;
        for (let j = 0; j < dim; j++) sums[assignments[i]][j] += data[i][j];
      }
      for (let c = 0; c < effK; c++) {
        if (counts[c] > 0) centroids[c] = sums[c].map((s) => s / counts[c]);
      }
      if (!changed && iter > 0) break;
    }
    let wcss = 0;
    for (let i = 0; i < data.length; i++) wcss += euclideanDist(data[i], centroids[assignments[i]]) ** 2;
    if (!best || wcss < best.wcss) best = { assignments, centroids, wcss };
  }
  return best;
}

function getSegmentLabels(k) {
  if (k === 1) return ['Seluruh Pelanggan'];
  const base = ['Pelanggan Utama (Champions)', 'Pelanggan Setia', 'Pelanggan Reguler', 'Pelanggan Potensial', 'Pelanggan Baru'];
  if (k - 1 <= base.length) return [...base.slice(0, k - 1), 'Pelanggan Berisiko (At Risk)'];
  const extra = [];
  for (let i = 0; i < k - 1 - base.length; i++) extra.push(`Segmen Tambahan ${i + 1}`);
  return [...base, ...extra, 'Pelanggan Berisiko (At Risk)'];
}

/* ============================================================
   COLUMN AUTO-DETECTION
   ============================================================ */
const FIELD_HINTS = {
  customerId: ['customerid', 'customer_id', 'idpelanggan', 'id_pelanggan', 'customer', 'pelanggan'],
  date: ['invoicedate', 'orderdate', 'order_date', 'date', 'tanggal', 'tanggaltransaksi', 'tgl'],
  invoiceId: ['invoiceno', 'invoice_no', 'orderid', 'order_id', 'notransaksi', 'no_transaksi', 'transactionid', 'nofaktur', 'no_faktur'],
  product: ['description', 'product', 'productname', 'namaproduk', 'nama_produk', 'produk', 'item'],
  quantity: ['quantity', 'qty', 'jumlah', 'jumlahbarang', 'jml'],
  price: ['unitprice', 'unit_price', 'price', 'harga', 'hargasatuan', 'harga_satuan'],
};
function autoDetectMapping(headers) {
  const norm = (s) => String(s).toLowerCase().replace(/[\s\-_.]/g, '');
  const mapping = {};
  for (const field of Object.keys(FIELD_HINTS)) {
    const hints = FIELD_HINTS[field];
    const found = headers.find((h) => hints.includes(norm(h)));
    mapping[field] = found || '';
  }
  return mapping;
}

/* ============================================================
   SMALL UI PRIMITIVES
   ============================================================ */
function Ticket({ children, style = {} }) {
  return (
    <div className="ticket" style={style}>
      {children}
    </div>
  );
}

function Kpi({ label, value, sub, accent }) {
  return (
    <Ticket>
      <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: C.muted, fontFamily: 'Inter, sans-serif' }}>
        {label}
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 26, fontWeight: 600, color: accent || C.text, marginTop: 6, lineHeight: 1.1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </Ticket>
  );
}

function SectionTitle({ eyebrow, title, desc }) {
  return (
    <div style={{ marginBottom: 20 }}>
      {eyebrow && (
        <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: C.amber, letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 6 }}>
          {eyebrow}
        </div>
      )}
      <h2 style={{ fontFamily: '"Fraunces", serif', fontSize: 26, fontWeight: 600, color: C.text, margin: 0 }}>{title}</h2>
      {desc && <p style={{ color: C.muted, fontSize: 14, marginTop: 8, maxWidth: 720, lineHeight: 1.6 }}>{desc}</p>}
    </div>
  );
}

const tooltipStyle = {
  background: C.surfaceAlt,
  border: `1px solid ${C.border}`,
  borderRadius: 6,
  fontFamily: '"IBM Plex Mono", monospace',
  fontSize: 12,
  color: C.text,
};

/* ============================================================
   MAIN APP
   ============================================================ */
export default function RetailKMeansDashboard() {
  const [stage, setStage] = useState('upload'); // upload | mapping | app
  const [fileName, setFileName] = useState('');
  const [headers, setHeaders] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [mapping, setMapping] = useState({});
  const [parseError, setParseError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  const [kValue, setKValue] = useState(3);
  const [elbowData, setElbowData] = useState(null);
  const [elbowLoading, setElbowLoading] = useState(false);
  const [clusterResult, setClusterResult] = useState(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [axisX, setAxisX] = useState('recency');
  const [axisY, setAxisY] = useState('monetary');

  /* ---------- FILE HANDLING ---------- */
  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setParseError('');
    setFileName(file.name);
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const hdrs = results.meta.fields || [];
        if (!hdrs.length || !results.data.length) {
          setParseError('File CSV tidak terbaca atau kosong. Pastikan baris pertama berisi nama kolom.');
          return;
        }
        setHeaders(hdrs);
        setRawData(results.data);
        setMapping(autoDetectMapping(hdrs));
        setStage('mapping');
      },
      error: () => setParseError('Gagal membaca file. Pastikan format file adalah .csv'),
    });
  };

  const resetAll = () => {
    setStage('upload');
    setFileName('');
    setHeaders([]);
    setRawData([]);
    setMapping({});
    setElbowData(null);
    setClusterResult(null);
    setActiveTab('overview');
  };

  /* ---------- DATA PROCESSING ---------- */
  const transactions = useMemo(() => {
    if (stage !== 'app') return [];
    const { customerId, date, quantity, price, invoiceId, product } = mapping;
    const out = [];
    for (const row of rawData) {
      const cust = row[customerId];
      const qty = Number(row[quantity]);
      const prc = Number(row[price]);
      const d = parseDate(row[date]);
      if (cust == null || cust === '' || !d || !isFinite(qty) || !isFinite(prc)) continue;
      if (qty <= 0 || prc <= 0) continue;
      out.push({
        customerId: String(cust),
        date: d,
        amount: qty * prc,
        invoiceId: invoiceId ? row[invoiceId] : null,
        product: product ? String(row[product]) : null,
      });
    }
    return out;
  }, [rawData, mapping, stage]);

  const customers = useMemo(() => {
    if (!transactions.length) return [];
    let maxDate = transactions[0].date;
    for (const t of transactions) if (t.date > maxDate) maxDate = t.date;
    const refDate = new Date(maxDate.getTime());
    refDate.setDate(refDate.getDate() + 1);

    const map = new Map();
    for (const t of transactions) {
      if (!map.has(t.customerId)) {
        map.set(t.customerId, { customerId: t.customerId, lastDate: t.date, invoices: new Set(), txCount: 0, monetary: 0 });
      }
      const c = map.get(t.customerId);
      if (t.date > c.lastDate) c.lastDate = t.date;
      if (t.invoiceId != null && t.invoiceId !== '') c.invoices.add(t.invoiceId);
      c.txCount++;
      c.monetary += t.amount;
    }
    const hasInvoice = !!mapping.invoiceId;
    return Array.from(map.values()).map((c) => ({
      customerId: c.customerId,
      recency: Math.max(0, Math.round((refDate - c.lastDate) / 86400000)),
      frequency: hasInvoice ? Math.max(c.invoices.size, 1) : c.txCount,
      monetary: Math.round(c.monetary),
    }));
  }, [transactions, mapping.invoiceId]);

  const normalizedFeatures = useMemo(() => {
    if (!customers.length) return [];
    const cols = ['recency', 'frequency', 'monetary'];
    const mins = {}, maxs = {};
    cols.forEach((col) => {
      const vals = customers.map((c) => c[col]);
      mins[col] = Math.min(...vals);
      maxs[col] = Math.max(...vals);
    });
    return customers.map((c) =>
      cols.map((col) => {
        const range = maxs[col] - mins[col];
        return range === 0 ? 0 : (c[col] - mins[col]) / range;
      })
    );
  }, [customers]);

  const overview = useMemo(() => {
    if (!transactions.length) return null;
    const totalRevenue = transactions.reduce((s, t) => s + t.amount, 0);
    const invoiceSet = new Set();
    const customerSet = new Set();
    const productMap = new Map();
    const monthMap = new Map();
    transactions.forEach((t) => {
      if (t.invoiceId != null && t.invoiceId !== '') invoiceSet.add(t.invoiceId);
      customerSet.add(t.customerId);
      const pname = t.product || 'Tidak diketahui';
      productMap.set(pname, (productMap.get(pname) || 0) + t.amount);
      const ym = `${t.date.getFullYear()}-${String(t.date.getMonth() + 1).padStart(2, '0')}`;
      monthMap.set(ym, (monthMap.get(ym) || 0) + t.amount);
    });
    const topProducts = Array.from(productMap.entries())
      .map(([name, value]) => ({ name: name.length > 18 ? name.slice(0, 18) + '…' : name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
    const monthly = Array.from(monthMap.entries())
      .map(([month, value]) => ({ month, value }))
      .sort((a, b) => a.month.localeCompare(b.month));
    return {
      totalRevenue,
      totalOrders: invoiceSet.size || transactions.length,
      totalCustomers: customerSet.size,
      totalProducts: productMap.size,
      avgOrder: totalRevenue / (invoiceSet.size || transactions.length),
      topProducts,
      monthly,
    };
  }, [transactions]);

  /* ---------- CLUSTERING ACTIONS ---------- */
  const runElbow = useCallback(() => {
    if (!normalizedFeatures.length) return;
    setElbowLoading(true);
    setTimeout(() => {
      const maxK = Math.min(8, normalizedFeatures.length);
      const results = [];
      for (let k = 1; k <= maxK; k++) {
        const r = runKMeans(normalizedFeatures, k, 50, 3);
        results.push({ k, wcss: Math.round(r.wcss * 100) / 100 });
      }
      setElbowData(results);
      setElbowLoading(false);
    }, 30);
  }, [normalizedFeatures]);

  const runClustering = useCallback(() => {
    if (!normalizedFeatures.length) return;
    setClusterLoading(true);
    setTimeout(() => {
      const r = runKMeans(normalizedFeatures, kValue, 80, 6);
      setClusterResult(r);
      setClusterLoading(false);
    }, 30);
  }, [normalizedFeatures, kValue]);

  const clusterSummary = useMemo(() => {
    if (!clusterResult || !clusterResult.assignments.length) return null;
    const k = Math.min(kValue, normalizedFeatures.length);
    const groups = Array.from({ length: k }, () => ({ count: 0, sumR: 0, sumF: 0, sumM: 0 }));
    customers.forEach((cust, i) => {
      const c = clusterResult.assignments[i];
      groups[c].count++;
      groups[c].sumR += cust.recency;
      groups[c].sumF += cust.frequency;
      groups[c].sumM += cust.monetary;
    });
    const stats = groups.map((g, idx) => ({
      cluster: idx,
      count: g.count,
      avgR: g.count ? g.sumR / g.count : 0,
      avgF: g.count ? g.sumF / g.count : 0,
      avgM: g.count ? g.sumM / g.count : 0,
    }));
    const maxR = Math.max(...stats.map((s) => s.avgR), 1);
    const maxF = Math.max(...stats.map((s) => s.avgF), 1);
    const maxM = Math.max(...stats.map((s) => s.avgM), 1);
    const scored = stats.map((s) => ({ ...s, score: s.avgF / maxF + s.avgM / maxM - s.avgR / maxR }));
    const sorted = [...scored].sort((a, b) => b.score - a.score);
    const labels = getSegmentLabels(k);
    const labelMap = {};
    sorted.forEach((s, rank) => (labelMap[s.cluster] = labels[rank]));
    return scored.map((s) => ({ ...s, label: labelMap[s.cluster] })).sort((a, b) => a.cluster - b.cluster);
  }, [clusterResult, customers, kValue, normalizedFeatures.length]);

  const scatterData = useMemo(() => {
    if (!clusterResult || !clusterResult.assignments.length) return [];
    return customers.map((c, i) => ({ ...c, cluster: clusterResult.assignments[i] }));
  }, [clusterResult, customers]);

  const pieData = useMemo(() => {
    if (!clusterSummary) return [];
    return clusterSummary.map((s) => ({ name: s.label, value: s.count, cluster: s.cluster }));
  }, [clusterSummary]);

  const downloadResults = () => {
    if (!clusterResult || !clusterSummary) return;
    const labelMap = {};
    clusterSummary.forEach((s) => (labelMap[s.cluster] = s.label));
    const rows = customers.map((c, i) => ({
      CustomerID: c.customerId,
      Recency: c.recency,
      Frequency: c.frequency,
      Monetary: c.monetary,
      Cluster: clusterResult.assignments[i],
      Segmen: labelMap[clusterResult.assignments[i]],
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hasil_segmentasi_kmeans.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ============================================================
     RENDER
     ============================================================ */
  return (
    <div style={{ minHeight: '100vh', background: C.bg, color: C.text, fontFamily: 'Inter, sans-serif' }}>
      <GlobalStyle />

      {stage === 'upload' && <UploadScreen onFile={handleFile} error={parseError} />}

      {stage === 'mapping' && (
        <MappingScreen
          headers={headers}
          rawData={rawData}
          mapping={mapping}
          setMapping={setMapping}
          fileName={fileName}
          onBack={resetAll}
          onConfirm={() => setStage('app')}
        />
      )}

      {stage === 'app' && (
        <div style={{ display: 'flex', minHeight: '100vh' }}>
          <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} fileName={fileName} onReset={resetAll} rowCount={transactions.length} />
          <main style={{ flex: 1, padding: '32px 40px', minWidth: 0 }}>
            {activeTab === 'overview' && <OverviewTab overview={overview} />}
            {activeTab === 'segmentation' && (
              <SegmentationTab
                customerCount={customers.length}
                kValue={kValue}
                setKValue={setKValue}
                elbowData={elbowData}
                elbowLoading={elbowLoading}
                runElbow={runElbow}
                clusterResult={clusterResult}
                clusterLoading={clusterLoading}
                runClustering={runClustering}
                clusterSummary={clusterSummary}
                scatterData={scatterData}
                pieData={pieData}
                axisX={axisX}
                axisY={axisY}
                setAxisX={setAxisX}
                setAxisY={setAxisY}
                downloadResults={downloadResults}
              />
            )}
          </main>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   GLOBAL STYLE
   ============================================================ */
function GlobalStyle() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

      * { box-sizing: border-box; }
      body, html { margin: 0; padding: 0; }

      .ticket {
        position: relative;
        background: ${C.surface};
        border: 1px solid ${C.border};
        border-radius: 6px;
        padding: 18px 20px;
      }
      .ticket::before, .ticket::after {
        content: '';
        position: absolute;
        width: 14px; height: 14px;
        background: ${C.bg};
        border-radius: 50%;
        top: 50%;
        transform: translateY(-50%);
      }
      .ticket::before { left: -8px; }
      .ticket::after { right: -8px; }

      .nav-btn {
        display: flex; align-items: center; gap: 10px;
        width: 100%; text-align: left;
        padding: 11px 14px; border-radius: 6px;
        background: transparent; border: none; cursor: pointer;
        color: ${C.muted}; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500;
        transition: background 0.15s, color 0.15s;
      }
      .nav-btn:hover { background: ${C.surfaceAlt}; color: ${C.text}; }
      .nav-btn.active { background: ${C.surfaceAlt}; color: ${C.amber}; }

      .btn-primary {
        background: ${C.amber}; color: #1A1306; border: none; border-radius: 6px;
        padding: 11px 22px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600;
        letter-spacing: 0.04em; cursor: pointer; transition: opacity 0.15s, transform 0.1s;
      }
      .btn-primary:hover { opacity: 0.88; }
      .btn-primary:active { transform: scale(0.98); }
      .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

      .btn-secondary {
        background: transparent; color: ${C.text}; border: 1px solid ${C.border}; border-radius: 6px;
        padding: 10px 20px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500;
        cursor: pointer; transition: border-color 0.15s, color 0.15s;
      }
      .btn-secondary:hover { border-color: ${C.amber}; color: ${C.amber}; }

      select, input[type=number] {
        background: ${C.surfaceAlt}; color: ${C.text}; border: 1px solid ${C.border};
        border-radius: 5px; padding: 8px 10px; font-family: 'IBM Plex Mono', monospace; font-size: 13px;
      }
      select:focus, input:focus { outline: 2px solid ${C.amber}; outline-offset: 1px; }

      table { width: 100%; border-collapse: collapse; font-size: 13px; }
      th { text-align: left; padding: 10px 14px; color: ${C.muted}; font-weight: 500; font-size: 11px;
           text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid ${C.border};
           font-family: 'Inter', sans-serif; }
      td { padding: 10px 14px; border-bottom: 1px solid ${C.border}; font-family: 'IBM Plex Mono', monospace;
           color: ${C.text}; font-size: 13px; }
      tr:last-child td { border-bottom: none; }

      input[type=file]::-webkit-file-upload-button {
        background: ${C.amber}; color: #1A1306; border: none; border-radius: 5px;
        padding: 9px 16px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; cursor: pointer; margin-right: 12px;
      }

      ::-webkit-scrollbar { width: 10px; height: 10px; }
      ::-webkit-scrollbar-track { background: ${C.bg}; }
      ::-webkit-scrollbar-thumb { background: ${C.surfaceAlt}; border-radius: 5px; }
    `}</style>
  );
}

/* ============================================================
   UPLOAD SCREEN
   ============================================================ */
function UploadScreen({ onFile, error }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
      <div style={{ maxWidth: 620, width: '100%' }}>
        <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: C.amber, letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: 14 }}>
          Sistem Analisis Data Penjualan Retail
        </div>
        <h1 style={{ fontFamily: '"Fraunces", serif', fontSize: 44, fontWeight: 600, lineHeight: 1.15, margin: 0, marginBottom: 16 }}>
          Segmentasi pelanggan berbasis <span style={{ color: C.amber }}>K-Means</span>
        </h1>
        <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.7, marginBottom: 32, maxWidth: 520 }}>
          Unggah data transaksi penjualan (.csv) untuk melihat ringkasan penjualan dan menjalankan
          segmentasi pelanggan berdasarkan analisis RFM (Recency, Frequency, Monetary) menggunakan
          algoritma K-Means.
        </p>

        <div className="ticket" style={{ padding: 28, borderStyle: 'dashed' }}>
          <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 13, color: C.muted, marginBottom: 14 }}>
            01 — UNGGAH FILE CSV
          </div>
          <input type="file" accept=".csv" onChange={onFile} style={{ color: C.muted, fontSize: 13 }} />
          {error && <div style={{ color: C.rose, fontSize: 13, marginTop: 14 }}>{error}</div>}
          <div style={{ marginTop: 18, paddingTop: 18, borderTop: `1px solid ${C.border}`, fontSize: 12, color: C.muted, lineHeight: 1.7 }}>
            Format kolom yang didukung (nama bebas — akan dipetakan di langkah berikutnya):
            <br />
            ID Pelanggan, Tanggal Transaksi, Jumlah (Qty), Harga Satuan, ID Transaksi/Invoice (opsional), Nama Produk (opsional).
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   MAPPING SCREEN
   ============================================================ */
const MAPPING_FIELDS = [
  { key: 'customerId', label: 'ID Pelanggan', required: true, desc: 'Identitas unik pelanggan' },
  { key: 'date', label: 'Tanggal Transaksi', required: true, desc: 'Digunakan untuk menghitung Recency' },
  { key: 'quantity', label: 'Jumlah (Qty)', required: true, desc: 'Jumlah unit terjual' },
  { key: 'price', label: 'Harga Satuan', required: true, desc: 'Harga per unit' },
  { key: 'invoiceId', label: 'ID Transaksi / Invoice', required: false, desc: 'Untuk menghitung Frequency (jika kosong, dihitung per baris)' },
  { key: 'product', label: 'Nama Produk', required: false, desc: 'Untuk ringkasan produk terlaris' },
];

function MappingScreen({ headers, rawData, mapping, setMapping, fileName, onBack, onConfirm }) {
  const requiredOk = MAPPING_FIELDS.filter((f) => f.required).every((f) => mapping[f.key]);
  const preview = rawData.slice(0, 5);

  return (
    <div style={{ minHeight: '100vh', padding: '40px 48px', maxWidth: 1000, margin: '0 auto' }}>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: C.amber, letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: 10 }}>
        02 — PEMETAAN KOLOM
      </div>
      <h1 style={{ fontFamily: '"Fraunces", serif', fontSize: 32, fontWeight: 600, margin: 0, marginBottom: 8 }}>{fileName}</h1>
      <p style={{ color: C.muted, fontSize: 14, marginBottom: 28 }}>
        {rawData.length.toLocaleString('id-ID')} baris terdeteksi. Sesuaikan kolom CSV kamu dengan kebutuhan analisis di bawah.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 32 }}>
        {MAPPING_FIELDS.map((f) => (
          <div key={f.key} className="ticket">
            <label style={{ display: 'block', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
              {f.label} {f.required && <span style={{ color: C.amber }}>*</span>}
            </label>
            <div style={{ fontSize: 12, color: C.muted, marginBottom: 10 }}>{f.desc}</div>
            <select
              value={mapping[f.key] || ''}
              onChange={(e) => setMapping((m) => ({ ...m, [f.key]: e.target.value }))}
              style={{ width: '100%' }}
            >
              <option value="">{f.required ? '— pilih kolom —' : '(tidak digunakan)'}</option>
              {headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <div style={{ marginBottom: 32, overflowX: 'auto' }} className="ticket">
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 10, fontFamily: '"IBM Plex Mono", monospace' }}>PRATINJAU DATA (5 BARIS)</div>
        <table>
          <thead>
            <tr>
              {headers.slice(0, 6).map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i}>
                {headers.slice(0, 6).map((h) => (
                  <td key={h}>{String(row[h] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', gap: 12 }}>
        <button className="btn-secondary" onClick={onBack}>
          ← Ganti File
        </button>
        <button className="btn-primary" disabled={!requiredOk} onClick={onConfirm}>
          Proses & Lanjutkan ke Dashboard →
        </button>
      </div>
      {!requiredOk && <div style={{ color: C.rose, fontSize: 12, marginTop: 10 }}>Lengkapi kolom wajib (*) sebelum melanjutkan.</div>}
    </div>
  );
}

/* ============================================================
   SIDEBAR
   ============================================================ */
function Sidebar({ activeTab, setActiveTab, fileName, onReset, rowCount }) {
  return (
    <aside style={{ width: 240, background: C.surface, borderRight: `1px solid ${C.border}`, padding: '28px 18px', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ padding: '0 6px', marginBottom: 28 }}>
        <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 11, color: C.amber, letterSpacing: '0.2em', textTransform: 'uppercase' }}>
          Retail Analytics
        </div>
        <div style={{ fontFamily: '"Fraunces", serif', fontSize: 19, fontWeight: 600, marginTop: 4 }}>K-Means Dashboard</div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 28 }}>
        <button className={`nav-btn ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
          📊 Ringkasan Penjualan
        </button>
        <button className={`nav-btn ${activeTab === 'segmentation' ? 'active' : ''}`} onClick={() => setActiveTab('segmentation')}>
          🎯 Segmentasi Pelanggan
        </button>
      </nav>

      <div style={{ marginTop: 'auto' }}>
        <div className="ticket" style={{ padding: 14, marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>File aktif</div>
          <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, wordBreak: 'break-all', marginBottom: 6 }}>{fileName}</div>
          <div style={{ fontSize: 11, color: C.muted }}>{rowCount.toLocaleString('id-ID')} transaksi valid</div>
        </div>
        <button className="btn-secondary" style={{ width: '100%' }} onClick={onReset}>
          Unggah File Lain
        </button>
      </div>
    </aside>
  );
}

/* ============================================================
   OVERVIEW TAB
   ============================================================ */
function OverviewTab({ overview }) {
  if (!overview) {
    return (
      <div className="ticket" style={{ padding: 24 }}>
        Tidak ada data transaksi valid yang dapat diolah. Periksa kembali pemetaan kolom (terutama Tanggal, Jumlah, dan Harga Satuan harus berupa angka/tanggal yang valid).
      </div>
    );
  }
  return (
    <div>
      <SectionTitle
        eyebrow="Dashboard"
        title="Ringkasan Penjualan"
        desc="Gambaran umum performa penjualan berdasarkan data transaksi yang diunggah."
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 18, marginBottom: 32 }}>
        <Kpi label="Total Pendapatan" value={fmtIDR(overview.totalRevenue)} accent={C.amber} />
        <Kpi label="Total Transaksi" value={fmtNum(overview.totalOrders)} />
        <Kpi label="Total Pelanggan" value={fmtNum(overview.totalCustomers)} />
        <Kpi label="Rata-rata / Transaksi" value={fmtIDR(overview.avgOrder)} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 18 }}>
        <div className="ticket" style={{ padding: 22 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 16, fontFamily: '"IBM Plex Mono", monospace', letterSpacing: '0.08em' }}>
            TREN PENDAPATAN BULANAN
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={overview.monthly}>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
              <XAxis dataKey="month" stroke={C.muted} fontSize={11} fontFamily="IBM Plex Mono, monospace" />
              <YAxis stroke={C.muted} fontSize={11} fontFamily="IBM Plex Mono, monospace" tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(0)}jt` : v)} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtIDR(v)} />
              <Line type="monotone" dataKey="value" stroke={C.amber} strokeWidth={2.5} dot={{ r: 3, fill: C.amber }} name="Pendapatan" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="ticket" style={{ padding: 22 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 16, fontFamily: '"IBM Plex Mono", monospace', letterSpacing: '0.08em' }}>
            PRODUK TERLARIS (BY REVENUE)
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={overview.topProducts} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" stroke={C.muted} fontSize={10} fontFamily="IBM Plex Mono, monospace" tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(0)}jt` : v)} />
              <YAxis type="category" dataKey="name" stroke={C.muted} fontSize={10} fontFamily="IBM Plex Mono, monospace" width={100} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtIDR(v)} />
              <Bar dataKey="value" fill={C.teal} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   SEGMENTATION TAB
   ============================================================ */
const AXIS_OPTIONS = [
  { key: 'recency', label: 'Recency (hari)' },
  { key: 'frequency', label: 'Frequency (kali transaksi)' },
  { key: 'monetary', label: 'Monetary (Rp)' },
];

function SegmentationTab(props) {
  const {
    customerCount, kValue, setKValue, elbowData, elbowLoading, runElbow,
    clusterResult, clusterLoading, runClustering, clusterSummary,
    scatterData, pieData, axisX, axisY, setAxisX, setAxisY, downloadResults,
  } = props;

  if (customerCount === 0) {
    return (
      <div className="ticket" style={{ padding: 24 }}>
        Tidak ada data pelanggan yang dapat dianalisis. Periksa kembali pemetaan kolom pada langkah sebelumnya.
      </div>
    );
  }

  const maxK = Math.min(8, customerCount);

  return (
    <div>
      <SectionTitle
        eyebrow="K-Means Clustering"
        title="Segmentasi Pelanggan (RFM)"
        desc={`Setiap pelanggan (total ${customerCount.toLocaleString('id-ID')}) diukur berdasarkan tiga variabel: Recency (jarak hari sejak transaksi terakhir), Frequency (jumlah transaksi), dan Monetary (total nilai pembelian). Ketiga variabel dinormalisasi (min-max scaling) sebelum dikelompokkan menggunakan algoritma K-Means.`}
      />

      {/* STEP 1: ELBOW */}
      <div className="ticket" style={{ padding: 22, marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <div>
            <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: C.amber, letterSpacing: '0.1em', marginBottom: 4 }}>
              LANGKAH 1
            </div>
            <div style={{ fontFamily: '"Fraunces", serif', fontSize: 19, fontWeight: 600 }}>Tentukan Jumlah Cluster (Elbow Method)</div>
            <div style={{ fontSize: 13, color: C.muted, marginTop: 4, maxWidth: 560 }}>
              Hitung Within-Cluster Sum of Squares (WCSS) untuk k = 1 sampai {maxK}. Titik "siku" pada grafik
              menunjukkan jumlah cluster yang optimal secara statistik.
            </div>
          </div>
          <button className="btn-primary" onClick={runElbow} disabled={elbowLoading}>
            {elbowLoading ? 'Menghitung…' : 'Hitung Elbow Method'}
          </button>
        </div>
        {elbowData && (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={elbowData}>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
              <XAxis dataKey="k" stroke={C.muted} fontSize={11} fontFamily="IBM Plex Mono, monospace" label={{ value: 'k (jumlah cluster)', position: 'insideBottom', offset: -2, fill: C.muted, fontSize: 11 }} />
              <YAxis stroke={C.muted} fontSize={11} fontFamily="IBM Plex Mono, monospace" label={{ value: 'WCSS', angle: -90, position: 'insideLeft', fill: C.muted, fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="wcss" stroke={C.amber} strokeWidth={2.5} dot={{ r: 4, fill: C.amber }} name="WCSS" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* STEP 2: RUN KMEANS */}
      <div className="ticket" style={{ padding: 22, marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16, marginBottom: 8 }}>
          <div>
            <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 12, color: C.amber, letterSpacing: '0.1em', marginBottom: 4 }}>
              LANGKAH 2
            </div>
            <div style={{ fontFamily: '"Fraunces", serif', fontSize: 19, fontWeight: 600 }}>Jalankan K-Means</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <label style={{ fontSize: 13, color: C.muted, display: 'flex', alignItems: 'center', gap: 8 }}>
              Jumlah cluster (k):
              <input
                type="number"
                min={2}
                max={maxK}
                value={kValue}
                onChange={(e) => setKValue(Math.max(2, Math.min(maxK, Number(e.target.value) || 2)))}
                style={{ width: 64 }}
              />
            </label>
            <button className="btn-primary" onClick={runClustering} disabled={clusterLoading}>
              {clusterLoading ? 'Memproses…' : 'Jalankan K-Means'}
            </button>
          </div>
        </div>
      </div>

      {/* RESULTS */}
      {clusterSummary && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 18, marginBottom: 24 }}>
            <div className="ticket" style={{ padding: 22 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
                <div style={{ fontSize: 12, color: C.muted, fontFamily: '"IBM Plex Mono", monospace', letterSpacing: '0.08em' }}>
                  VISUALISASI CLUSTER
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <AxisSelect label="Sumbu X" value={axisX} onChange={setAxisX} />
                  <AxisSelect label="Sumbu Y" value={axisY} onChange={setAxisY} />
                </div>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ left: 10, right: 10, bottom: 10 }}>
                  <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey={axisX}
                    name={AXIS_OPTIONS.find((a) => a.key === axisX).label}
                    stroke={C.muted}
                    fontSize={11}
                    fontFamily="IBM Plex Mono, monospace"
                    tickFormatter={(v) => (axisX === 'monetary' && v >= 1e6 ? `${(v / 1e6).toFixed(0)}jt` : v)}
                  />
                  <YAxis
                    type="number"
                    dataKey={axisY}
                    name={AXIS_OPTIONS.find((a) => a.key === axisY).label}
                    stroke={C.muted}
                    fontSize={11}
                    fontFamily="IBM Plex Mono, monospace"
                    tickFormatter={(v) => (axisY === 'monetary' && v >= 1e6 ? `${(v / 1e6).toFixed(0)}jt` : v)}
                  />
                  <ZAxis range={[40, 41]} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} formatter={(v, n) => (n === 'monetary' ? fmtIDR(v) : v)} />
                  {clusterSummary.map((s) => (
                    <Scatter
                      key={s.cluster}
                      name={s.label}
                      data={scatterData.filter((d) => d.cluster === s.cluster)}
                      fill={CLUSTER_COLORS[s.cluster % CLUSTER_COLORS.length]}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'Inter, sans-serif' }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="ticket" style={{ padding: 22 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 16, fontFamily: '"IBM Plex Mono", monospace', letterSpacing: '0.08em' }}>
                PROPORSI SEGMEN
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ percent }) => `${(percent * 100).toFixed(0)}%`}>
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                {pieData.map((entry, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: C.muted }}>
                    <span style={{ width: 10, height: 10, borderRadius: '50%', background: CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length], display: 'inline-block' }} />
                    {entry.name}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SUMMARY TABLE */}
          <div className="ticket" style={{ padding: 22, marginBottom: 24, overflowX: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
              <div style={{ fontSize: 12, color: C.muted, fontFamily: '"IBM Plex Mono", monospace', letterSpacing: '0.08em' }}>
                KARAKTERISTIK CLUSTER
              </div>
              <button className="btn-secondary" onClick={downloadResults}>
                ⬇ Unduh Hasil (CSV)
              </button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Cluster</th>
                  <th>Segmen</th>
                  <th>Jumlah Pelanggan</th>
                  <th>Rata-rata Recency (hari)</th>
                  <th>Rata-rata Frequency</th>
                  <th>Rata-rata Monetary</th>
                </tr>
              </thead>
              <tbody>
                {clusterSummary.map((s) => (
                  <tr key={s.cluster}>
                    <td>
                      <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: CLUSTER_COLORS[s.cluster % CLUSTER_COLORS.length], marginRight: 8 }} />
                      {s.cluster}
                    </td>
                    <td style={{ fontFamily: 'Inter, sans-serif', fontWeight: 500 }}>{s.label}</td>
                    <td>{fmtNum(s.count)}</td>
                    <td>{fmtNum(s.avgR, 1)}</td>
                    <td>{fmtNum(s.avgF, 1)}</td>
                    <td>{fmtIDR(s.avgM)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="ticket" style={{ padding: '16px 22px', fontSize: 12, color: C.muted, lineHeight: 1.7 }}>
            <strong style={{ color: C.text }}>Catatan interpretasi:</strong> label segmen di atas dihasilkan secara
            heuristik berdasarkan peringkat skor gabungan (Frequency + Monetary − Recency, dinormalisasi) tiap
            cluster. Untuk paper, kamu dapat menyesuaikan penamaan segmen sesuai konteks bisnis dan
            menjelaskan parameter K-Means yang digunakan (jarak Euclidean, inisialisasi K-Means++, iterasi
            maksimum 80, dengan 6 kali percobaan acak untuk memperoleh hasil terbaik / WCSS terendah).
          </div>
        </>
      )}
    </div>
  );
}

function AxisSelect({ label, value, onChange }) {
  return (
    <label style={{ fontSize: 11, color: C.muted, display: 'flex', alignItems: 'center', gap: 6 }}>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {AXIS_OPTIONS.map((o) => (
          <option key={o.key} value={o.key}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
