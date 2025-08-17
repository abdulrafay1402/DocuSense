# DocuSense 📚

> AI-powered document analysis and semantic search platform

DocuSense transforms your PDF documents into an intelligent, searchable knowledge base using advanced AI technologies. Upload documents, ask natural language questions, and get AI-powered explanations for any passage.

## ✨ Features

- **📄 Smart PDF Processing**: Extract and chunk text from PDF documents with advanced parsing
- **🔍 Semantic Search**: Find relevant passages using natural language queries
- **🤖 AI Explanations**: Get contextual explanations powered by Google's Gemini AI
- **📊 Search History**: Track and revisit your previous searches and insights
- **⚡ Fast Indexing**: Efficient document indexing using FAISS vector search
- **🎨 Modern UI**: Clean, dark-themed interface with smooth animations

## 🚀 Live Demo

Visit the live application: [docu-sense-ai.streamlit.app/](https://docu-sense-ai.streamlit.app/)

*Upload PDFs → Ask Questions → Get AI Insights*

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS styling
- **AI/ML**: 
  - Google Gemini 1.5 Pro for explanations
  - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
  - FAISS for vector similarity search
- **Document Processing**: PyMuPDF for PDF text extraction
- **Backend**: Python with async/await support

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API Key
- 2GB+ RAM (for embedding models)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/abdulrafay1402/docusense.git
cd docusense
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Get your Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into your `.env` file

## 🚀 Usage

### Local Development
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Upload Documents** 📤
   - Navigate to the Upload page
   - Select one or more PDF files
   - Configure processing settings (optional)
   - Click "Process Documents"

2. **Search & Explore** 🔍
   - Go to the Search page
   - Ask natural language questions
   - Review ranked results by relevance
   - Click "Explain" for AI-powered insights

3. **Review History** 📜
   - Check the History page for past searches
   - Review previous explanations
   - Monitor system statistics

## 📁 Project Structure

```
docusense/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## ⚙️ Configuration

### Processing Settings
- **Chunk Size**: Text chunk size for indexing (300-2000 characters)
- **Overlap**: Overlap between chunks for context preservation
- **Results Count**: Number of search results to display

### Performance Tuning
- Larger chunk sizes: Better context, slower processing
- Higher overlap: Better context continuity, more storage
- More results: Broader search coverage, longer response times

## 🔒 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for AI explanations |

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `GEMINI_API_KEY` to Streamlit secrets
4. Deploy!

### Local Production
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🎯 Example Use Cases

- **Research Papers**: "What methodology was used in this study?"
- **Legal Documents**: "What are the key terms in this contract?"
- **Technical Manuals**: "How do I configure this feature?"
- **Reports**: "What were the main findings on page 5?"

## 🔧 Troubleshooting

### Common Issues

**PDF Processing Errors**
- Ensure PDFs contain extractable text (not just images)
- Try re-saving PDFs if extraction fails

**API Key Issues**
- Verify your Gemini API key is correct
- Check API quotas and billing in Google Cloud Console

**Memory Issues**
- Reduce chunk sizes for large documents
- Process fewer documents simultaneously

**Slow Performance**
- Use smaller embedding models for faster processing
- Reduce the number of search results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for academic and institutional use. Credit the developers if reused or modified for deployment.


## 👨‍💻 Author

**Abdul Rafay**
- GitHub: [@abdulrafay1402](https://github.com/abdulrafay1402)
- LinkedIn: [Abdul Rafay Imran](https://linkedin.com/in/abdulrafay-imran)
- Email: abdulrafay14021997@gmail.com

*Software Engineering Student at FAST-NUCES, Karachi, Pakistan*

## 🙏 Acknowledgments

- Google for the Gemini AI API
- Streamlit team for the amazing framework
- Sentence Transformers for embeddings
- FAISS team for efficient vector search
- PyMuPDF for PDF processing capabilities

## 📈 Roadmap

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Batch processing for multiple documents
- [ ] Export search results and explanations
- [ ] Advanced filtering and sorting options
- [ ] Multi-language support
- [ ] Integration with cloud storage services

---

<div align="center">
  <p>Made with ❤️ by Abdul Rafay</p>
  <p>© 2024-2025 Abdul Rafay. All rights reserved.</p>
</div>
