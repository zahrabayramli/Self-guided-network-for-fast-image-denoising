class SubNetwork(nn.Module):
    def __init__(self, g, m, c_in, c_k, c_out, sub_network_type):
        super(SubNetwork, self).__init__()

        self.act = nn.ReLU()
    
        self.conv_first = nn.ModuleList()
        self.conv_first.append(nn.Conv2d(c_in, c_k, kernel_size=3, stride=1, padding=1, bias=False))

        if sub_network_type == 'MIDDLE':
          self.conv_first.append(nn.Conv2d(c_k // 2 * 3, c_k, kernel_size=3, stride=1, padding=1, bias=False))
        
        if sub_network_type == 'BOTTOM':
          self.res_block = None
          self.conv_last1 = nn.ModuleList()
          for i in range(m//2):
            self.conv_last1.append(nn.Conv2d(c_k, c_k, kernel_size=3, stride=1, padding=1, bias=False))

          self.conv_last2 = nn.ModuleList()
          self.conv_last2.append(nn.Conv2d(c_k // 2 * 3, c_k, kernel_size=3, stride=1, padding=1, bias=False))

          for i in range(m - m//2):
            self.conv_last2.append(nn.Conv2d(c_k, c_k, kernel_size=3, stride=1, padding=1, bias=False))
          
          self.conv_last2.append(nn.Conv2d(c_k, c_out, kernel_size=3, stride=1, padding=1, bias=False))

        else:
          self.res_block = nn.ModuleList()
          for i in range(g):
            self.res_block.append(nn.Conv2d(c_k, c_k, kernel_size=3, stride=1, padding=1, bias=False))
          
          self.conv_last = nn.Conv2d(c_k, c_k, kernel_size=3, stride=1, padding=1, bias=False)
          
    def forward(self, x, upper_features=None):

      x = self.act(self.conv_first[0](x))

      if len(self.conv_first) == 2:
        x = torch.concat((x, F.pixel_shuffle(upper_features, 2)), 1)
        x = self.act(self.conv_first[1](x))

      if self.res_block != None:
        y = x
        for i, l in enumerate(self.res_block):
          y = l(y)
          if i != len(self.res_block) - 1:
            y = self.act(y)
        x = self.act(torch.add(x, y))
        x = self.act(self.conv_last(x))

      else:
        for l in self.conv_last1:
          x = self.act(l(x))

        x = torch.concat((x, F.pixel_shuffle(upper_features, 2)), 1)
        x = self.act(self.conv_last2[0](x))

        for l in self.conv_last2[1:-1]:
          x = self.act(l(x))

        x = self.conv_last2[-1](x)
        
      return x

class SGN(nn.Module):
    def __init__(self, opt):
        super(SGN, self).__init__()
        
        c_in = opt.in_channels
        c_0 = opt.start_channels
        c_out = opt.out_channels
        g = 3
        m = opt.m_block
        self.K = 2

        self.bottom = SubNetwork(g, m, c_in, c_0, c_out, 'BOTTOM')

        c_k = c_0
        self.middle = nn.ModuleList()
        for i in range(self.K):
          c_k *= 2
          c_in *= 4
          self.middle.append(SubNetwork(g, m, c_in, c_k, c_out, 'MIDDLE'))

        c_k *= 2
        c_in *= 4
        self.top = SubNetwork(g, m, c_in, c_k, c_out, 'TOP')

    def forward(self, x):
      l = [x]
      for i in range(self.K + 1):
        l.append(F.pixel_unshuffle(l[-1], 2))
      
      upper_features = self.top(l[-1])
      i = -2
      for middle in reversed(self.middle):
        upper_features = middle(l[i], upper_features)
        i -= 1
      
      x = self.bottom(x, upper_features)

      return x