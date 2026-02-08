"""
Test suite for the refactored spectra module.
"""

# Import necessary modules
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from astropy import units as u
from astropy.table import Table
from unite.spectra import Spectrum, NIRSpecSpectrum, Spectra


class TestSpectrum:
    """Test the generic Spectrum class"""

    def test_initialization_basic(self):
        """Test basic spectrum initialization with units"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        assert spectrum.name == "test_spectrum"
        assert spectrum.redshift_initial == 0.0
        assert not spectrum.fixed
        np.testing.assert_array_equal(spectrum.wave.value, wave.value)
        np.testing.assert_array_equal(spectrum.flux.value, flux.value)
        np.testing.assert_array_equal(spectrum.err.value, err.value)

    def test_initialization_no_units_error(self):
        """Test that initialization fails without units"""
        wave = np.array([1.0, 2.0, 3.0])  # No units
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        with pytest.raises(ValueError, match="wave must have units"):
            Spectrum(
                name="test_spectrum",
                wave=wave,
                flux=flux,
                err=err,
            )

    def test_initialization_with_parameters(self):
        """Test spectrum initialization with custom parameters"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
            redshift_initial=0.5,
            fixed=True,
        )
        
        assert spectrum.redshift_initial == 0.5
        assert spectrum.fixed

    def test_bin_edge_calculation(self):
        """Test automatic bin edge calculation"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        # Check that bin edges were calculated
        assert len(spectrum.low) == len(wave)
        assert len(spectrum.high) == len(wave)
        assert spectrum.low[0].value < spectrum.wave[0].value
        assert spectrum.high[-1].value > spectrum.wave[-1].value

    def test_nan_masking(self):
        """Test that NaN values are properly masked"""
        wave = np.array([1.0, 2.0, 3.0, 4.0]) * u.micron
        flux = np.array([10.0, np.nan, 30.0, 40.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, np.nan, 4.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        # Should have 2 valid points (indices 0 and 3)
        assert len(spectrum.wave) == 2
        np.testing.assert_array_equal(spectrum.wave.value, [1.0, 4.0])
        np.testing.assert_array_equal(spectrum.flux.value, [10.0, 40.0])
        np.testing.assert_array_equal(spectrum.err.value, [1.0, 4.0])

    def test_coverage(self):
        """Test coverage method"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        low = np.array([0.5, 1.5, 2.5]) * u.micron
        high = np.array([1.5, 2.5, 3.5]) * u.micron
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
            low=low,
            high=high,
        )
        
        # Test partial coverage
        coverage = spectrum.coverage(1.0, 2.0, partial=True)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(coverage, expected)
        
        # Test full coverage
        coverage = spectrum.coverage(1.6, 2.4, partial=False)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(coverage, expected)

    def test_continuum_mask(self):
        """Test setting continuum mask"""
        wave = np.array([1.0, 2.0, 3.0, 4.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0, 40.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0, 4.0]) * u.Unit('erg s-1 cm-2 AA-1')
        low = np.array([0.5, 1.5, 2.5, 3.5]) * u.micron
        high = np.array([1.5, 2.5, 3.5, 4.5]) * u.micron
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
            low=low,
            high=high,
        )
        
        # Set continuum regions [1.6, 2.4] and [3.6, 4.4]
        continuum_regions = [(1.6, 2.4), (3.6, 4.4)]
        spectrum.set_continuum_mask(continuum_regions)
        
        # Should keep indices 1 and 3
        assert len(spectrum.wave) == 2
        np.testing.assert_array_equal(spectrum.wave.value, [2.0, 4.0])
        np.testing.assert_array_equal(spectrum.flux.value, [20.0, 40.0])

    def test_call_method(self):
        """Test __call__ method"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test_spectrum",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        low, wave_out, high, flux_out, err_out = spectrum()
        np.testing.assert_array_equal(wave_out.value, wave.value)
        np.testing.assert_array_equal(flux_out.value, flux.value)
        np.testing.assert_array_equal(err_out.value, err.value)


class TestNIRSpecSpectrum:
    """Test the NIRSpecSpectrum class"""

    def test_initialization(self):
        """Test NIRSpec spectrum initialization"""
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = NIRSpecSpectrum(
            disperser="G395M",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        assert spectrum.name == "G395M"
        assert hasattr(spectrum, 'lsf')
        assert hasattr(spectrum, 'offset')

    @patch('unite.spectra.Table.read')
    def test_from_file(self, mock_read):
        """Test creating NIRSpecSpectrum from file"""
        # Mock the spectrum file data
        mock_table = MagicMock()
        mock_table.__getitem__.side_effect = lambda key: {
            'wave': np.array([1.0, 2.0, 3.0]) * u.micron,
            'flux': np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            'err': np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1'),
        }[key]
        mock_read.return_value = mock_table
        
        spectrum = NIRSpecSpectrum.from_file(
            disperser="G395M",
            spec_file="test.fits",
            redshift_initial=0.1,
        )
        
        assert spectrum.name == "G395M"
        assert spectrum.redshift_initial == 0.1
        mock_read.assert_called_once_with("test.fits", 'SPEC1D')

    @patch('unite.spectra.urllib.request.urlopen')
    @patch('unite.spectra.Table.read')
    def test_from_dja(self, mock_read, mock_urlopen):
        """Test creating NIRSpecSpectrum from DJA"""
        # Mock the spectrum file data
        mock_table = MagicMock()
        mock_table.__getitem__.side_effect = lambda key: {
            'wave': np.array([1.0, 2.0, 3.0]) * u.micron,
            'flux': np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            'err': np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1'),
        }[key]
        mock_read.return_value = mock_table
        
        spectrum = NIRSpecSpectrum.from_dja(
            mask="test-mask",
            srcid=12345,
            disperser="G395M",
            redshift_initial=0.1,
        )
        
        assert spectrum.name == "G395M"
        assert spectrum.redshift_initial == 0.1
        mock_urlopen.assert_called_once()


class TestSpectra:
    """Test the Spectra collection class"""

    def setup_method(self):
        """Set up test spectra"""
        self.spectrum1 = Spectrum(
            name="spectrum1",
            wave=np.array([1.0, 2.0, 3.0]) * u.micron,
            flux=np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            err=np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            redshift_initial=0.1,
            fixed=True,
        )
        
        self.spectrum2 = Spectrum(
            name="spectrum2",
            wave=np.array([3.5, 4.5, 5.5]) * u.micron,
            flux=np.array([35.0, 45.0, 55.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            err=np.array([3.5, 4.5, 5.5]) * u.Unit('erg s-1 cm-2 AA-1'),
            redshift_initial=0.1,
            fixed=False,
        )

    def test_initialization(self):
        """Test Spectra initialization"""
        spectra = Spectra([self.spectrum1, self.spectrum2])
        
        assert len(spectra.spectra) == 2
        assert spectra.names == ["spectrum1", "spectrum2"]
        assert spectra.fixed == [True, False]
        assert spectra.redshift_initial == 0.1

    def test_initialization_with_parameters(self):
        """Test Spectra initialization with custom parameters"""
        spectra = Spectra(
            [self.spectrum1, self.spectrum2],
            redshift_initial=0.2,
            fixed=[False, True],
        )
        
        assert spectra.redshift_initial == 0.2
        assert spectra.fixed == [False, True]
        # Check that individual spectra were updated
        assert not spectra.spectra[0].fixed
        assert spectra.spectra[1].fixed

    @patch('unite.spectra.urllib.request.urlopen')
    @patch('unite.spectra.Table.read')
    def test_from_dja(self, mock_read, mock_urlopen):
        """Test creating Spectra from DJA"""
        # Mock the catalog data
        catalog_data = Table({
            'mask': ['test-mask', 'test-mask'],
            'srcid': [12345, 12345],
            'grating': ['G395M', 'PRISM'],
            'grade': [2, 1],
            'z': [0.1, 0.1],
            'zfit': [0.1, 0.1],
        })
        
        # Mock spectrum data
        spectrum_data = MagicMock()
        spectrum_data.__getitem__.side_effect = lambda key: {
            'wave': np.array([1.0, 2.0, 3.0]) * u.micron,
            'flux': np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            'err': np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1'),
        }[key]
        
        # Set up the mock to return catalog first, then spectrum data
        mock_read.side_effect = [catalog_data, spectrum_data, spectrum_data]
        
        spectra = Spectra.from_dja(
            mask="test-mask",
            srcid=12345,
        )
        
        assert len(spectra.spectra) == 2
        assert spectra.redshift_initial == 0.1

    def test_from_dja_no_spectra_found(self):
        """Test from_dja when no spectra are found"""
        with patch('unite.spectra.urllib.request.urlopen'), \
             patch('unite.spectra.Table.read') as mock_read:
            
            # Mock empty catalog
            catalog_data = Table({
                'mask': [],
                'srcid': [],
                'grating': [],
                'grade': [],
                'z': [],
                'zfit': [],
            })
            mock_read.return_value = catalog_data
            
            with pytest.raises(ValueError, match="No spectra found"):
                Spectra.from_dja(mask="nonexistent", srcid=99999)


class TestIntegration:
    """Integration tests for the refactored module"""

    def test_end_to_end_workflow(self):
        """Test a complete workflow with the refactored classes"""
        # Create some test spectra
        spectrum1 = Spectrum(
            name="G395M",
            wave=np.array([1.0, 2.0, 3.0, 4.0]) * u.micron,
            flux=np.array([10.0, 20.0, 30.0, 40.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            err=np.array([1.0, 2.0, 3.0, 4.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            redshift_initial=0.1,
            fixed=True,
        )
        
        spectrum2 = Spectrum(
            name="PRISM",
            wave=np.array([1.5, 2.5, 3.5, 4.5]) * u.micron,
            flux=np.array([15.0, 25.0, 35.0, 45.0]) * u.Unit('erg s-1 cm-2 AA-1'),
            err=np.array([1.5, 2.5, 3.5, 4.5]) * u.Unit('erg s-1 cm-2 AA-1'),
            redshift_initial=0.1,
            fixed=False,
        )
        
        # Create spectra collection
        spectra = Spectra([spectrum1, spectrum2])
        
        # Test restriction using continuum regions
        continuum_regions = [(1.5, 2.5), (3.5, 4.5)]
        spectra.restrict(continuum_regions)
        
        # Both spectra should have data in these regions
        assert len(spectra.spectra) == 2
        
        # Check that data was properly restricted
        for spectrum in spectra.spectra:
            assert len(spectrum.wave) > 0

    def test_units_consistency(self):
        """Test that units are handled consistently"""
        spectrum = Spectrum(
            name="test",
            wave=np.array([1.0, 2.0, 3.0]) * u.angstrom,
            flux=np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 Hz-1'),
            err=np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 Hz-1'),
        )
        
        assert spectrum.λ_unit == u.angstrom
        assert spectrum.fλ_unit == u.Unit('erg s-1 cm-2 Hz-1')

    def test_backwards_compatibility(self):
        """Test that the refactored code maintains key functionality"""
        # This tests that the main use cases still work
        
        # Create a spectrum
        wave = np.array([1.0, 2.0, 3.0]) * u.micron
        flux = np.array([10.0, 20.0, 30.0]) * u.Unit('erg s-1 cm-2 AA-1')
        err = np.array([1.0, 2.0, 3.0]) * u.Unit('erg s-1 cm-2 AA-1')
        
        spectrum = Spectrum(
            name="test",
            wave=wave,
            flux=flux,
            err=err,
        )
        
        # Test the __call__ method works
        low, wave_out, high, flux_out, err_out = spectrum()
        
        assert len(wave_out) == 3
        assert len(flux_out) == 3
        assert len(err_out) == 3


# Test configuration for pytest
def pytest_configure():
    """Configure pytest for this test module"""
    pass


if __name__ == "__main__":
    pytest.main([__file__])
