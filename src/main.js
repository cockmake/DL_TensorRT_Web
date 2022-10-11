import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

import axios from "axios"
axios.defaults.baseURL = "http://127.0.0.1:5000"
// axios.defaults.baseURL = " http://10.32.47.121:5000"
createApp(App).mount('#app')
