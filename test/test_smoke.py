"""Smoke tests for scientific fixes + Flask route map (no full training)."""
import os
import sys
import math
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


class TestQuantumBasics(unittest.TestCase):
    def test_endianness_x0_is_msb(self):
        import torch
        from core.quantum import quantum_circuit
        qc = quantum_circuit(3, device='cpu')
        qc.x(0)
        probs = qc.probabilities().real.flatten()
        self.assertEqual(int(torch.argmax(probs)), 4)  # |100⟩ with MSB=q0

    def test_zz_matches_msb_gates(self):
        from core.quantum import quantum_circuit
        qc = quantum_circuit(2, device='cpu')
        qc.x(0)
        # |10⟩ → Z0Z1 eigenvalue = (-1)*(+1) = -1
        self.assertAlmostEqual(qc.expectation_ZZ(0, 1), -1.0, places=5)

    def test_qaoa_params_have_grad(self):
        import torch
        from core.models import Hybrid_QNN
        m = Hybrid_QNN(num_qubits=3, num_layers=1, num_classes=10, use_qaoa=True, nisq_error_rate=0.0)
        x = torch.randn(2, 1, 28, 28)
        loss = m(x).sum()
        loss.backward()
        self.assertIsNotNone(m.angles.grad)
        self.assertIsNotNone(m.gammas.grad)
        self.assertIsNotNone(m.betas.grad)
        self.assertGreater(float(m.gammas.grad.abs().sum()), 0.0)

    def test_fidelity_uses_prev_global_not_client_sum(self):
        import torch
        from core.fl.aggregation import indirect_quantum_aggregation
        d = 4
        s1 = torch.zeros(d, dtype=torch.cfloat); s1[0] = 1
        s2 = torch.zeros(d, dtype=torch.cfloat); s2[1] = 1
        s3 = torch.zeros(d, dtype=torch.cfloat); s3[0] = 1
        # prev global = s1 → clients s1,s2 should weight high/low vs s1
        sd = {'w': torch.tensor([1.0]), 'b': torch.tensor([0.0])}
        locals_ = [ {'w': torch.tensor([float(i)]), 'b': torch.tensor([0.0])} for i in (1, 2, 3)]
        _, weights = indirect_quantum_aggregation(locals_, [s1, s2, s3], s1, 2, 'cpu')
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(weights[0], weights[2], places=5)


class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from webapp import create_app
        cls.client = create_app().test_client()

    def test_pages_ok(self):
        for path in ['/', '/about', '/contact', '/choose', '/compare', '/simulate', '/simulate_own']:
            rv = self.client.get(path)
            self.assertEqual(rv.status_code, 200, msg=path)

    def test_about_mentions_central_server(self):
        rv = self.client.get('/about')
        text = rv.data.decode('utf-8').lower()
        self.assertIn('central server', text)
        self.assertIn('not a peer-to-peer', text)

    def test_home_not_claim_ring_only_params(self):
        rv = self.client.get('/')
        text = rv.data.decode('utf-8').lower()
        self.assertIn('central server', text)
        self.assertNotIn('travel the ring', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
